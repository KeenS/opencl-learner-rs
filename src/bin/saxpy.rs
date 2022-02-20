use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_float, CL_TRUE};
use opencl3::Result;

static SOURCE: &str = include_str!("../saxpy.cl");
static KERNEL_NAME: &str = "saxpy";

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
    const LENGTH: usize = 256;
    let mut px: [cl_float; LENGTH] = [0.0; LENGTH];
    let mut py: [cl_float; LENGTH] = [0.0; LENGTH];
    for i in 0..LENGTH {
        px[i] = i as cl_float;
        py[i] = (LENGTH - 1 - i) as cl_float;
    }
    println!("px: {px:?}");
    println!("py: {py:?}");

    const NITEMS: usize = 512;
    let buf_x = Buffer::<cl_float>::create(
        &context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        NITEMS,
        px.as_mut_ptr() as *mut _,
    )?;
    let buf_y = Buffer::<cl_float>::create(
        &context,
        CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
        NITEMS,
        py.as_mut_ptr() as *mut _,
    )?;
    let global_work_size = NITEMS;
    let a: cl_float = 2.0;
    let kernel_event = ExecuteKernel::new(&kernel)
        .set_arg(&buf_x)
        .set_arg(&buf_y)
        .set_arg(&a)
        .set_global_work_size(global_work_size)
        .set_local_work_size(64)
        .enqueue_nd_range(&queue)?;
    kernel_event.wait()?;
    let event = queue
        .enqueue_read_buffer(&buf_y, CL_TRUE, 0, &mut py, &[])
        .expect("enqueue_map_buffer");
    event.wait()?;
    println!("{py:?}");
    Ok(())
}
