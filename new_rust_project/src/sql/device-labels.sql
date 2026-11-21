CREATE TABLE IF NOT EXISTS device_labels (
    device_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL
);
INSERT INTO device_labels (device_id, display_name) VALUES
('planar_mosfet', 'Планарный MOSFET (bulk)'),
('fd_soi', 'FD-SOI MOSFET'),
('finfet', 'FinFET'),
('gaa_nanosheet', 'GAA nanosheet'),
('gaa_nanowire', 'GAA nanowire'),
('tfet', 'TFET'),
('hemt', 'HEMT'),
('memristor', 'Мемристор'),
('memtransistor', 'Мемтранзистор'),
('cad_custom', 'CAD-транзистор');
