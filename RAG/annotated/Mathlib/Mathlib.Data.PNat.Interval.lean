instance instLocallyFiniteOrder : LocallyFiniteOrder ℕ+ := Subtype.instLocallyFiniteOrder _


theorem Icc_eq_finset_subtype : Icc a b = (Icc (a : ℕ) b).subtype fun n : ℕ => 0 < n :=
  rfl


theorem Ico_eq_finset_subtype : Ico a b = (Ico (a : ℕ) b).subtype fun n : ℕ => 0 < n :=
  rfl


theorem Ioc_eq_finset_subtype : Ioc a b = (Ioc (a : ℕ) b).subtype fun n : ℕ => 0 < n :=
  rfl


theorem Ioo_eq_finset_subtype : Ioo a b = (Ioo (a : ℕ) b).subtype fun n : ℕ => 0 < n :=
  rfl


theorem uIcc_eq_finset_subtype : uIcc a b = (uIcc (a : ℕ) b).subtype fun n : ℕ => 0 < n := rfl


theorem map_subtype_embedding_Icc : (Icc a b).map (Embedding.subtype _) = Icc ↑a ↑b :=
  Finset.map_subtype_embedding_Icc _ _ _ fun _c _ _x hx _ hc _ => hc.trans_le hx


theorem map_subtype_embedding_Ico : (Ico a b).map (Embedding.subtype _) = Ico ↑a ↑b :=
  Finset.map_subtype_embedding_Ico _ _ _ fun _c _ _x hx _ hc _ => hc.trans_le hx


theorem map_subtype_embedding_Ioc : (Ioc a b).map (Embedding.subtype _) = Ioc ↑a ↑b :=
  Finset.map_subtype_embedding_Ioc _ _ _ fun _c _ _x hx _ hc _ => hc.trans_le hx


theorem map_subtype_embedding_Ioo : (Ioo a b).map (Embedding.subtype _) = Ioo ↑a ↑b :=
  Finset.map_subtype_embedding_Ioo _ _ _ fun _c _ _x hx _ hc _ => hc.trans_le hx


theorem map_subtype_embedding_uIcc : (uIcc a b).map (Embedding.subtype _) = uIcc ↑a ↑b :=
  map_subtype_embedding_Icc _ _


@[simp]
theorem card_Icc : #(Icc a b) = b + 1 - a := by
  /-
    a b : PNat
    ⊢ Eq (Finset.Icc a b).card (HSub.hSub (HAdd.hAdd (↑b) 1) ↑a)
  -/
  rw [← Nat.card_Icc, ← map_subtype_embedding_Icc, card_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_Ico : #(Ico a b) = b - a := by
  /-
    a b : PNat
    ⊢ Eq (Finset.Ico a b).card (HSub.hSub ↑b ↑a)
  -/
  rw [← Nat.card_Ico, ← map_subtype_embedding_Ico, card_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_Ioc : #(Ioc a b) = b - a := by
  /-
    a b : PNat
    ⊢ Eq (Finset.Ioc a b).card (HSub.hSub ↑b ↑a)
  -/
  rw [← Nat.card_Ioc, ← map_subtype_embedding_Ioc, card_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_Ioo : #(Ioo a b) = b - a - 1 := by
  /-
    a b : PNat
    ⊢ Eq (Finset.Ioo a b).card (HSub.hSub (HSub.hSub ↑b ↑a) 1)
  -/
  rw [← Nat.card_Ioo, ← map_subtype_embedding_Ioo, card_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_uIcc : #(uIcc a b) = (b - a : ℤ).natAbs + 1 := by
  /-
    a b : PNat
    ⊢ Eq (Finset.uIcc a b).card (HAdd.hAdd (HSub.hSub ↑↑b ↑↑a).natAbs 1)
  -/
  rw [← Nat.card_uIcc, ← map_subtype_embedding_uIcc, card_map]
  /-
    🎉 no goals
  -/

-- Porting note: `simpNF` says `simp` can prove this

theorem card_fintype_Icc : Fintype.card (Set.Icc a b) = b + 1 - a := by
  /-
    a b : PNat
    ⊢ Eq (Fintype.card ↑(Set.Icc a b)) (HSub.hSub (HAdd.hAdd (↑b) 1) ↑a)
  -/
  rw [← card_Icc, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/

-- Porting note: `simpNF` says `simp` can prove this

theorem card_fintype_Ico : Fintype.card (Set.Ico a b) = b - a := by
  /-
    a b : PNat
    ⊢ Eq (Fintype.card ↑(Set.Ico a b)) (HSub.hSub ↑b ↑a)
  -/
  rw [← card_Ico, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/

-- Porting note: `simpNF` says `simp` can prove this

theorem card_fintype_Ioc : Fintype.card (Set.Ioc a b) = b - a := by
  /-
    a b : PNat
    ⊢ Eq (Fintype.card ↑(Set.Ioc a b)) (HSub.hSub ↑b ↑a)
  -/
  rw [← card_Ioc, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/

-- Porting note: `simpNF` says `simp` can prove this

theorem card_fintype_Ioo : Fintype.card (Set.Ioo a b) = b - a - 1 := by
  /-
    a b : PNat
    ⊢ Eq (Fintype.card ↑(Set.Ioo a b)) (HSub.hSub (HSub.hSub ↑b ↑a) 1)
  -/
  rw [← card_Ioo, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/

-- Porting note: `simpNF` says `simp` can prove this

theorem card_fintype_uIcc : Fintype.card (Set.uIcc a b) = (b - a : ℤ).natAbs + 1 := by
  /-
    a b : PNat
    ⊢ Eq (Fintype.card ↑(Set.uIcc a b)) (HAdd.hAdd (HSub.hSub ↑↑b ↑↑a).natAbs 1)
  -/
  rw [← card_uIcc, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/


