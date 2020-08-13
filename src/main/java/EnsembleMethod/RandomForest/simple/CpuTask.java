package EnsembleMethod.RandomForest.simple;


import java.lang.management.ManagementFactory;
import java.text.DecimalFormat;
import java.util.TimerTask;

import com.opencsv.CSVWriter;
import com.sun.management.OperatingSystemMXBean;


public class CpuTask extends TimerTask {

	CSVWriter writer = null;
	DecimalFormat df = null;
	//System.out.println(df.format(PI));

	public  CpuTask(CSVWriter writer)
	{
		super();
		this.writer = writer;
		df = new DecimalFormat("###.##");
	}
	@Override
	public void run() {
		
		OperatingSystemMXBean osMBean = ManagementFactory.getPlatformMXBean(
                OperatingSystemMXBean.class);
		
		System.out.println("Number of active threads"+Thread.activeCount());
		System.out.println("Process Cpu Load:"+osMBean.getProcessCpuLoad() *100L);

		System.out.println("System Cpu load:"+osMBean.getSystemCpuLoad() *100L);
		System.out.println("Cpu Processing Time: "+osMBean.getProcessCpuTime()/1000000000);
		
		 // add data to csv 
        String[] data1 = {""+Thread.activeCount(), ""+df.format(osMBean.getProcessCpuLoad() *100), ""+df.format(osMBean.getSystemCpuLoad() *100), ""+df.format(osMBean.getProcessCpuTime()/1000000000)}; 
        writer.writeNext(data1); 
		
	}
}
