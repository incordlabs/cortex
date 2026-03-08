use tracel_xtask::prelude::*;

pub(crate) fn handle_command(
    mut args: DocCmdArgs,
    env: Environment,
    ctx: Context,
) -> anyhow::Result<()> {
    if args.get_command() == DocSubCommand::Build {
        args.exclude
            .extend(vec!["cortex-cuda".to_string(), "cortex-rocm".to_string()]);
    }

    // Execute documentation command on workspace
    base_commands::doc::handle_command(args.clone(), env, ctx)?;

    // Specific additional commands to build other docs
    if args.get_command() == DocSubCommand::Build {
        // cortex-dataset
        helpers::custom_crates_doc_build(
            vec!["cortex-dataset"],
            vec!["--all-features"],
            None,
            None,
            "All features",
        )?;
    }
    Ok(())
}
