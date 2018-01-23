function imshowRGY(A,B,lbls)
x = 6.25*(0:105);
u = 40*(-1.95:.195:1.95);
figure
iptsetpref('ImshowAxesVisible','on');
subplot(3,1,1)
imshow(cat(3,A,zeros(size(A)),zeros(size(A))),'XData',x,'YData',u)
xlabel('x (um)')
ylabel('kx (a.u.)')
set(gca,'YTickLabel',[])
title(lbls{1})
subplot(3,1,2)
imshow(cat(3,zeros(size(A)),B,zeros(size(A))),'XData',x,'YData',u)
xlabel('x (um)')
ylabel('kx (a.u.)')
set(gca,'YTickLabel',[])
title(lbls{2})
subplot(3,1,3)
imshow(cat(3,A,B,zeros(size(A))),'XData',x,'YData',u)
xlabel('x (um)')
ylabel('angle (a.u.)')
set(gca,'YTickLabel',[])
title('merged')
end