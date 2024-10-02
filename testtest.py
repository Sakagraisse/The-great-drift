import unittest
import torch
import numpy as np

from SIM import activate_torch_mps, create_initial_pop, create_storage, store_data, migration

class TestSimulationFunctions(unittest.TestCase):

    def setUp(self):
        self.group_size = 5
        self.number_groups = 3
        self.num_interactions = 10
        self.transfert_multiplier = 0.5
        self.x_i_value = 1.0
        self.choice = 0
        self.period = 5
        self.to_migrate = 2
        self.device = activate_torch_mps()

    def test_activate_torch_mps(self):
        device = activate_torch_mps()
        self.assertIn(device.type, ['cpu', 'mps'])

    def test_create_initial_pop(self):
        x_i, d_i, a_i, store_interaction, fitnessIN, fitnessOUT, fitnessToT = create_initial_pop(
            self.group_size, self.number_groups, self.num_interactions, self.transfert_multiplier, self.x_i_value, self.choice, self.device
        )
        self.assertEqual(x_i.shape, (self.number_groups, self.group_size))
        self.assertEqual(d_i.shape, (self.number_groups, self.group_size))
        self.assertEqual(a_i.shape, (self.number_groups, self.group_size))
        self.assertEqual(store_interaction.shape, (self.number_groups, self.group_size, self.num_interactions))
        self.assertEqual(fitnessIN.shape, (self.number_groups, self.group_size))
        self.assertEqual(fitnessOUT.shape, (self.number_groups, self.group_size))
        self.assertEqual(fitnessToT.shape, (self.number_groups, self.group_size))

    def test_create_storage(self):
        storage_x, storage_d, storage_a, storage_store_interaction, storage_fitnessIN, storage_fitnessOUT, storage_fitnessToT = create_storage(
            self.period, self.group_size, self.number_groups
        )
        self.assertEqual(storage_x.shape, (self.period, self.group_size * self.number_groups))
        self.assertEqual(storage_d.shape, (self.period, self.group_size * self.number_groups))
        self.assertEqual(storage_a.shape, (self.period, self.group_size * self.number_groups))
        self.assertEqual(storage_store_interaction.shape, (self.period, self.group_size * self.number_groups))
        self.assertEqual(storage_fitnessIN.shape, (self.period, self.group_size * self.number_groups))
        self.assertEqual(storage_fitnessOUT.shape, (self.period, self.group_size * self.number_groups))
        self.assertEqual(storage_fitnessToT.shape, (self.period, self.group_size * self.number_groups))

    def test_store_data(self):
        x_i, d_i, a_i, store_interaction, fitnessIN, fitnessOUT, fitnessToT = create_initial_pop(
            self.group_size, self.number_groups, self.num_interactions, self.transfert_multiplier, self.x_i_value, self.choice, self.device
        )
        storage_x, storage_d, storage_a, storage_store_interaction, storage_fitnessIN, storage_fitnessOUT, storage_fitnessToT = create_storage(
            self.period, self.group_size, self.number_groups
        )
        store_data(x_i, d_i, a_i, store_interaction, fitnessIN, fitnessOUT, fitnessToT,
                   storage_x, storage_d, storage_a, storage_store_interaction, storage_fitnessIN,
                   storage_fitnessOUT, storage_fitnessToT, 0)
        self.assertTrue(np.array_equal(storage_x[0], x_i.flatten().cpu().numpy()))
        self.assertTrue(np.array_equal(storage_d[0], d_i.flatten().cpu().numpy()))
        self.assertTrue(np.array_equal(storage_a[0], a_i.flatten().cpu().numpy()))
        #self.assertTrue(np.array_equal(storage_store_interaction[0], store_interaction.flatten().cpu().numpy()))
        self.assertTrue(np.array_equal(storage_fitnessIN[0], fitnessIN.flatten().cpu().numpy()))
        self.assertTrue(np.array_equal(storage_fitnessOUT[0], fitnessOUT.flatten().cpu().numpy()))
        self.assertTrue(np.array_equal(storage_fitnessToT[0], fitnessToT.flatten().cpu().numpy()))

    def test_migration(self):
        x_i, d_i, a_i, store_interaction, fitnessIN, fitnessOUT, fitnessToT = create_initial_pop(
            self.group_size, self.number_groups, self.num_interactions, self.transfert_multiplier, self.x_i_value, self.choice, self.device
        )
        migration(x_i, d_i, a_i, fitnessToT, fitnessOUT, fitnessIN, self.number_groups, self.group_size, self.to_migrate)
        self.assertEqual(x_i.shape, (self.number_groups, self.group_size))
        self.assertEqual(d_i.shape, (self.number_groups, self.group_size))
        self.assertEqual(a_i.shape, (self.number_groups, self.group_size))
        self.assertEqual(fitnessToT.shape, (self.number_groups, self.group_size))
        self.assertEqual(fitnessOUT.shape, (self.number_groups, self.group_size))
        self.assertEqual(fitnessIN.shape, (self.number_groups, self.group_size))

if __name__ == '__main__':
    unittest.main()