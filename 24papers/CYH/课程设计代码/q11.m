clc,clear,close all;
%% 第一步，根据有无缺陷分类，构建训练集。
%注：运行前在E盘建立两个文件夹，名为non_defective和visble
targetPath0 = 'E:\non_defective\';%无缺陷
targetPath1 = 'E:\visble\';%有缺陷
tic
for i =1:50
       fdname = sprintf('kos%02d',i);%文件夹名称
       fd = dir(fdname);
    for j = 1:(size(fd,1)-2)/2
       phname = fd(2*j+1).name;%原图片名称
       lbname = fd(2*j+2).name;%标签图片名称
       yfig0 = figure('Visible','off');
       I0 = imread([fdname,'\',phname]);
       imshow(I0);
       I1 = imread([fdname,'\',lbname]);
       if ~isempty(find(any(I1)==1, 1))%如果是有缺陷的
           saveas(yfig0,[targetPath1,'kos',num2str(i),phname],'jpg')%把原图保存到targetPath1
       else
           saveas(yfig0,[targetPath0,'kos',num2str(i),phname],'jpg')%把原图保存到targetPath0
       end
    end
end
toc









