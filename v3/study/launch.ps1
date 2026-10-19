# Start run_bg.bat outside the SSH session's job object, so it survives a disconnect.
$dir = "C:\Users\maik\pacing_v3\study"
$cmd = "$dir\run_bg.bat " + ($args -join " ")
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = $dir}
"return $($r.ReturnValue) pid $($r.ProcessId)"
