clear;

vib_normal   = readmatrix('./datas/vibration_normal.csv');
vib_abnormal = readmatrix('./datas/current_anomaly.csv');
cur_normal   = readmatrix('./datas/current_normal.csv');
cur_abnormal = readmatrix('./datas/current_anomaly.csv');

% vibration signal visualization
vib_normal_freq   = vib_normal(1,2:end);
vib_abnormal_freq = vib_abnormal(1,2:end); 
vib_normal_data   = vib_normal(2,2:end);
vib_abnormal_data = vib_abnormal(2,2:end);

% figure;
% subplot(2,1,1);
% plot(vib_normal_freq, vib_normal_data);
% xlabel("freq [Hz]");        ylabel("vibration");        title("normal vibration data");
% grid minor;
% subplot(2,1,2);
% plot(vib_abnormal_freq, vib_abnormal_data);
% xlabel("freq [Hz]");        ylabel("vibration");        title("abnormal vibration data");
% grid minor;

% electric current signal visualization
% cur_normal_freq   = cur_normal(1,2:end);
% cur_abnormal_freq = cur_abnormal(1,2:end); 
% cur_normal_data   = cur_normal(2,2:end);
% cur_abnormal_data = cur_abnormal(2,2:end);
% 
% figure;
% subplot(2,1,1);
% plot(cur_normal_freq, cur_normal_data);
% xlabel("freq [Hz]");        ylabel("currnet");        title("normal current data");
% grid minor;
% subplot(2,1,2);
% plot(cur_abnormal_freq, cur_abnormal_data);
% xlabel("freq [Hz]");        ylabel("currnet");        title("abnormal current data");
% grid minor;

t = linspace(0, 0.3, 512);
time_domain = ifft(vib_normal_data);

plot(t, time_domain);