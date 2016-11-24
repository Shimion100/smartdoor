package edu.muc.service;

import com.demo.index.TestPython;
import edu.muc.jxd.vo.YiCharacterDao;
import edu.muc.jxd.vo.YiCharcter;
import org.apache.commons.io.FileUtils;
import org.apache.log4j.Logger;

import java.io.File;
import java.io.IOException;

/**
 * Created by gwd on 2016/6/5.
 */
public class YiWenServices {
    private static String path=YiWenServices.class.getResource("/").getPath();
    private static File file_Path=new File(path);
    private static Logger logger= Logger.getLogger(YiWenServices.class.getName());

    public static String yiwenScan(File file, String altitude, String latitude){

        File yiwen=new File(file_Path.getPath()+File.separator+"YiCharacterPredict"+File.separator+"data"+File.separator+"src.png");
        logger.info("A file at "+file.getPath()+"will be operation by yiwen scanner.");
        try {
            FileUtils.copyFile(file,yiwen);
            logger.info("copy a file to "+yiwen.getPath());
        } catch (IOException e) {
            e.printStackTrace();
        }
        String location=file_Path.getPath()+File.separator+"YiCharacterPredict"+File.separator+"src"+File.separator+"Predict.py";
        logger.info("the python model is running.["+location+"]");
        String id= TestPython.execPython(file_Path.getPath()+File.separator+"YiCharacterPredict"+File.separator+"src"+File.separator+"Predict.py",null);
        YiCharcter charcter = new YiCharcter(altitude, latitude,id,yiwen.getPath());
        YiCharacterDao dao = new YiCharacterDao();
        dao.addYiCharacter(charcter);
        if(charcter!=null&&charcter.getMeaning()!=null){
            logger.info("the meaning result is :"+charcter.getMeaning());
            return  charcter.getMeaning();
        }
        else{
            logger.info("charter is null or charter.getMeaning() is null. Reset the result：请稍后在重试");
            return "请稍后在重试";
        }

    }

}
