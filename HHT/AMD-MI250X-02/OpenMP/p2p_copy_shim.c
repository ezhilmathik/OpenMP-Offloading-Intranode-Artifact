#define _GNU_SOURCE
#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

typedef hsa_status_t (HSA_API *copy_on_engine_fn_t)(
    void *,
    hsa_agent_t,
    const void *,
    hsa_agent_t,
    size_t,
    uint32_t,
    const hsa_signal_t *,
    hsa_signal_t,
    hsa_amd_sdma_engine_id_t,
    bool);

static copy_on_engine_fn_t get_real_copy(void)
{
    static copy_on_engine_fn_t fn = NULL;

    if (fn)
        return fn;

    fn = (copy_on_engine_fn_t)dlvsym(
        RTLD_NEXT,
        "hsa_amd_memory_async_copy_on_engine",
        "ROCR_1");

    if (!fn) {
        fn = (copy_on_engine_fn_t)dlsym(
            RTLD_NEXT,
            "hsa_amd_memory_async_copy_on_engine");
    }

    if (!fn) {
        fprintf(stderr,
                "P2P shim: cannot locate real copy function: %s\n",
                dlerror());
    }

    return fn;
}

static hsa_status_t pointer_info(
    const void *ptr,
    hsa_amd_pointer_info_t *info)
{
    memset(info, 0, sizeof(*info));
    info->size = sizeof(*info);

    return hsa_amd_pointer_info(
        ptr,
        info,
        NULL,
        NULL,
        NULL);
}

/*
 * For a two-GPU diagnostic, grant the remote agent access to the
 * allocation containing ptr.
 */
static hsa_status_t allow_remote_access(
    const void *ptr,
    hsa_agent_t remote_agent,
    const char *name)
{
    hsa_amd_pointer_info_t info;
    hsa_status_t status = pointer_info(ptr, &info);

    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr,
                "P2P shim: pointer_info(%s=%p) failed: %d\n",
                name, ptr, (int)status);
        return status;
    }

    if (info.type == HSA_EXT_POINTER_TYPE_UNKNOWN ||
        info.agentBaseAddress == NULL) {
        fprintf(stderr,
                "P2P shim: unknown allocation for %s=%p\n",
                name, ptr);
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    status = hsa_amd_agents_allow_access(
        1,
        &remote_agent,
        NULL,
        info.agentBaseAddress);

    if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr,
                "P2P shim: allow_access failed:"
                " %s_base=%p remote_agent=0x%lx status=%d\n",
                name,
                info.agentBaseAddress,
                (unsigned long)remote_agent.handle,
                (int)status);
    }

    return status;
}

hsa_status_t HSA_API hsa_amd_memory_async_copy_on_engine(
    void *dst,
    hsa_agent_t dst_agent,
    const void *src,
    hsa_agent_t src_agent,
    size_t size,
    uint32_t num_dep_signals,
    const hsa_signal_t *dep_signals,
    hsa_signal_t completion_signal,
    hsa_amd_sdma_engine_id_t requested_engine,
    bool force_copy_on_sdma)
{
    copy_on_engine_fn_t real_copy = get_real_copy();

    if (!real_copy)
        return HSA_STATUS_ERROR;

    /*
     * Preserve normal behavior for same-agent initialization and
     * runtime-internal copies.
     */
    if (dst_agent.handle == src_agent.handle) {
        return real_copy(
            dst,
            dst_agent,
            src,
            src_agent,
            size,
            num_dep_signals,
            dep_signals,
            completion_signal,
            requested_engine,
            force_copy_on_sdma);
    }

    /*
     * Both agents must be able to access both allocations.
     *
     * Destination owner already accesses dst, so grant src_agent
     * access to dst. Source owner already accesses src, so grant
     * dst_agent access to src.
     */
    hsa_status_t status =
        allow_remote_access(dst, src_agent, "dst");

    if (status != HSA_STATUS_SUCCESS)
        return status;

    status = allow_remote_access(src, dst_agent, "src");

    if (status != HSA_STATUS_SUCCESS)
        return status;

    uint32_t available_mask = 0;

    status = hsa_amd_memory_copy_engine_status(
        dst_agent,
        src_agent,
        &available_mask);

    if (status != HSA_STATUS_SUCCESS || available_mask == 0) {
        fprintf(stderr,
                "P2P shim: no available peer SDMA engine:"
                " status=%d mask=0x%x\n",
                (int)status,
                available_mask);
        return status;
    }

    uint32_t selected_engine;

    if (((uint32_t)requested_engine & available_mask) != 0) {
        selected_engine = (uint32_t)requested_engine;
    } else {
        /* Select the lowest valid single-bit engine ID. */
        selected_engine =
            available_mask & (~available_mask + 1u);
    }

#ifdef P2P_SHIM_VERBOSE
    fprintf(stderr,
            "P2P shim:"
            " dst_agent=0x%lx"
            " src_agent=0x%lx"
            " requested=0x%x"
            " available=0x%x"
            " selected=0x%x"
            " size=%zu\n",
            (unsigned long)dst_agent.handle,
            (unsigned long)src_agent.handle,
            (unsigned)requested_engine,
            available_mask,
            selected_engine,
            size);
#endif

    return real_copy(
        dst,
        dst_agent,
        src,
        src_agent,
        size,
        num_dep_signals,
        dep_signals,
        completion_signal,
        (hsa_amd_sdma_engine_id_t)selected_engine,
        true);
}
