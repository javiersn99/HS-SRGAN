function saveFigures(folder,name)
path=strcat(folder,name,"\");
mkdir(path)

exportgraphics(gcf, strcat(path,name,'.png'));
saveas(gcf, strcat(path,name,'.fig'));
saveas(gca, strcat(path,name,'.svg'));