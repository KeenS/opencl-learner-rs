use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_uint, CL_TRUE};
use opencl3::Result;
use std::ptr;

static SOURCE: &str = include_str!("../memset.cl");
static KERNEL_NAME: &str = "memset";

fn main() -> Result<()> {
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);
    let context = Context::from_device(&device).expect("Context::from_device failed");
    let queue = CommandQueue::create(&context, context.default_device(), 0)
        .expect("CommandQueue::create failed");
    let program = Program::create_and_build_from_source(&context, SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    ////////////////////////////////////////////////////////////////
    // Compute data

    // The input data
    const NITEMS: usize = 512;
    let buffer = Buffer::<cl_uint>::create(&context, CL_MEM_WRITE_ONLY, NITEMS, ptr::null_mut())?;
    let global_work_size = NITEMS;
    let kernel_event = ExecuteKernel::new(&kernel)
        .set_arg(&buffer)
        .set_global_work_size(global_work_size)
        .enqueue_nd_range(&queue)?;
    kernel_event.wait()?;
    let mut results = [0; NITEMS];
    let event = queue
        .enqueue_read_buffer(&buffer, CL_TRUE, 0, &mut results, &[])
        .expect("enqueue_map_buffer");
    event.wait()?;
    for (i, e) in results.iter().enumerate() {
        println!("{i}: {e}");
    }
    Ok(())
}
