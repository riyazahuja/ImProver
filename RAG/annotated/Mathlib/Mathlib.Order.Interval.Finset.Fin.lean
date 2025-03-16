@[simp, norm_cast]
theorem coe_sup : ↑(a ⊔ b) = (a ⊔ b : ℕ) := rfl


@[simp, norm_cast]
theorem coe_inf : ↑(a ⊓ b) = (a ⊓ b : ℕ) := rfl


@[simp, norm_cast]
theorem coe_max : ↑(max a b) = (max a b : ℕ) := rfl


@[simp, norm_cast]
theorem coe_min : ↑(min a b) = (min a b : ℕ) := rfl


instance instLocallyFiniteOrder : LocallyFiniteOrder (Fin n) :=
  OrderIso.locallyFiniteOrder Fin.orderIsoSubtype


instance instLocallyFiniteOrderBot : LocallyFiniteOrderBot (Fin n) :=
  OrderIso.locallyFiniteOrderBot Fin.orderIsoSubtype


instance instLocallyFiniteOrderTop : ∀ n, LocallyFiniteOrderTop (Fin n)
  | 0 => IsEmpty.toLocallyFiniteOrderTop
  | _ + 1 => inferInstance


theorem Icc_eq_finset_subtype : Icc a b = (Icc (a : ℕ) b).fin n :=
  rfl


theorem Ico_eq_finset_subtype : Ico a b = (Ico (a : ℕ) b).fin n :=
  rfl


theorem Ioc_eq_finset_subtype : Ioc a b = (Ioc (a : ℕ) b).fin n :=
  rfl


theorem Ioo_eq_finset_subtype : Ioo a b = (Ioo (a : ℕ) b).fin n :=
  rfl


theorem uIcc_eq_finset_subtype : uIcc a b = (uIcc (a : ℕ) b).fin n := rfl


@[simp]
theorem map_valEmbedding_Icc : (Icc a b).map Fin.valEmbedding = Icc ↑a ↑b := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Eq (Finset.map Fin.valEmbedding (Finset.Icc a b)) (Finset.Icc ↑a ↑b)
  -/
  simp [Icc_eq_finset_subtype, Finset.fin, Finset.map_map, Icc_filter_lt_of_lt_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_valEmbedding_Ico : (Ico a b).map Fin.valEmbedding = Ico ↑a ↑b := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Eq (Finset.map Fin.valEmbedding (Finset.Ico a b)) (Finset.Ico ↑a ↑b)
  -/
  simp [Ico_eq_finset_subtype, Finset.fin, Finset.map_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_valEmbedding_Ioc : (Ioc a b).map Fin.valEmbedding = Ioc ↑a ↑b := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Eq (Finset.map Fin.valEmbedding (Finset.Ioc a b)) (Finset.Ioc ↑a ↑b)
  -/
  simp [Ioc_eq_finset_subtype, Finset.fin, Finset.map_map, Ioc_filter_lt_of_lt_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_valEmbedding_Ioo : (Ioo a b).map Fin.valEmbedding = Ioo ↑a ↑b := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Eq (Finset.map Fin.valEmbedding (Finset.Ioo a b)) (Finset.Ioo ↑a ↑b)
  -/
  simp [Ioo_eq_finset_subtype, Finset.fin, Finset.map_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_subtype_embedding_uIcc : (uIcc a b).map valEmbedding = uIcc ↑a ↑b :=
  map_valEmbedding_Icc _ _


@[simp]
                                              /-
                                                n : Nat
                                                a b : Fin n
                                                ⊢ Eq (Finset.Icc a b).card (HSub.hSub (HAdd.hAdd (↑b) 1) ↑a)
                                              -/
lemma card_Icc : #(Icc a b) = b + 1 - a := by rw [← Nat.card_Icc, ← map_valEmbedding_Icc, card_map]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
                                          /-
                                            n : Nat
                                            a b : Fin n
                                            ⊢ Eq (Finset.Ico a b).card (HSub.hSub ↑b ↑a)
                                          -/
lemma card_Ico : #(Ico a b) = b - a := by rw [← Nat.card_Ico, ← map_valEmbedding_Ico, card_map]
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
                                          /-
                                            n : Nat
                                            a b : Fin n
                                            ⊢ Eq (Finset.Ioc a b).card (HSub.hSub ↑b ↑a)
                                          -/
lemma card_Ioc : #(Ioc a b) = b - a := by rw [← Nat.card_Ioc, ← map_valEmbedding_Ioc, card_map]
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
                                              /-
                                                n : Nat
                                                a b : Fin n
                                                ⊢ Eq (Finset.Ioo a b).card (HSub.hSub (HSub.hSub ↑b ↑a) 1)
                                              -/
lemma card_Ioo : #(Ioo a b) = b - a - 1 := by rw [← Nat.card_Ioo, ← map_valEmbedding_Ioo, card_map]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem card_uIcc : #(uIcc a b) = (b - a : ℤ).natAbs + 1 := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Eq (Finset.uIcc a b).card (HAdd.hAdd (HSub.hSub ↑↑b ↑↑a).natAbs 1)
  -/
  rw [← Nat.card_uIcc, ← map_subtype_embedding_uIcc, card_map]
  /-
    🎉 no goals
  -/


theorem card_fintypeIcc : Fintype.card (Set.Icc a b) = b + 1 - a := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Eq (Fintype.card ↑(Set.Icc a b)) (HSub.hSub (HAdd.hAdd (↑b) 1) ↑a)
  -/
  rw [← card_Icc, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/


theorem card_fintypeIco : Fintype.card (Set.Ico a b) = b - a := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Eq (Fintype.card ↑(Set.Ico a b)) (HSub.hSub ↑b ↑a)
  -/
  rw [← card_Ico, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/


theorem card_fintypeIoc : Fintype.card (Set.Ioc a b) = b - a := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Eq (Fintype.card ↑(Set.Ioc a b)) (HSub.hSub ↑b ↑a)
  -/
  rw [← card_Ioc, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/


theorem card_fintypeIoo : Fintype.card (Set.Ioo a b) = b - a - 1 := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Eq (Fintype.card ↑(Set.Ioo a b)) (HSub.hSub (HSub.hSub ↑b ↑a) 1)
  -/
  rw [← card_Ioo, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/


theorem card_fintype_uIcc : Fintype.card (Set.uIcc a b) = (b - a : ℤ).natAbs + 1 := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Eq (Fintype.card ↑(Set.uIcc a b)) (HAdd.hAdd (HSub.hSub ↑↑b ↑↑a).natAbs 1)
  -/
  rw [← card_uIcc, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/


theorem Ici_eq_finset_subtype : Ici a = (Icc (a : ℕ) n).fin n := by
  /-
    n : Nat
    a : Fin n
    ⊢ Eq (Finset.Ici a) (Finset.fin n (Finset.Icc (↑a) n))
  -/
  ext
  /-
    case h
    n : Nat
    a a✝ : Fin n
    ⊢ Iff (Membership.mem (Finset.Ici a) a✝) (Membership.mem (Finset.fin n (Finset …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Ioi_eq_finset_subtype : Ioi a = (Ioc (a : ℕ) n).fin n := by
  /-
    n : Nat
    a : Fin n
    ⊢ Eq (Finset.Ioi a) (Finset.fin n (Finset.Ioc (↑a) n))
  -/
  ext
  /-
    case h
    n : Nat
    a a✝ : Fin n
    ⊢ Iff (Membership.mem (Finset.Ioi a) a✝) (Membership.mem (Finset.fin n (Finset …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Iic_eq_finset_subtype : Iic b = (Iic (b : ℕ)).fin n :=
  rfl


theorem Iio_eq_finset_subtype : Iio b = (Iio (b : ℕ)).fin n :=
  rfl


@[simp]
theorem map_valEmbedding_Ici : (Ici a).map Fin.valEmbedding = Icc ↑a (n - 1) := by
  /-
    n : Nat
    a : Fin n
    ⊢ Eq (Finset.map Fin.valEmbedding (Finset.Ici a)) (Finset.Icc (↑a) (HSub.hSub  …
  -/
  ext x
  /-
    case h
    n : Nat
    a : Fin n
    x : Nat
    ⊢ Iff (Membership.mem (Finset.map Fin.valEmbedding (Finset.Ici a)) x) (Members …
  -/
  simp only [exists_prop, Embedding.coe_subtype, mem_Ici, mem_map, mem_Icc]
  /-
    case h
    n : Nat
    a : Fin n
    x : Nat
    ⊢ Iff (Exists fun a_1 => And (LE.le a a_1) (Eq (Fin.valEmbedding a_1) x)) (And …
  -/
  constructor
    /-
      case h.mp
      n : Nat
      a : Fin n
      x : Nat
      ⊢ (Exists fun a_1 => And (LE.le a a_1) (Eq (Fin.valEmbedding a_1) x)) → And (L …
    -/
  · rintro ⟨x, hx, rfl⟩
    /-
      case h.mp.intro.intro
      n : Nat
      a x : Fin n
      hx : LE.le a x
      ⊢ And (LE.le (↑a) (Fin.valEmbedding x)) (LE.le (Fin.valEmbedding x) (HSub.hSub …
    -/
    exact ⟨hx, Nat.le_sub_of_add_le <| x.2⟩
    /-
      🎉 no goals
    -/
  /-
    case h.mpr
    n : Nat
    a : Fin n
    x : Nat
    ⊢ And (LE.le (↑a) x) (LE.le x (HSub.hSub n 1)) → Exists fun a_2 => And (LE.le  …
  -/
  cases n
    /-
      case h.mpr.zero
      x : Nat
      a : Fin 0
      ⊢ And (LE.le (↑a) x) (LE.le x (HSub.hSub 0 1)) → Exists fun a_2 => And (LE.le  …
    -/
  · exact Fin.elim0 a
    /-
      🎉 no goals
    -/
    /-
      case h.mpr.succ
      x n✝ : Nat
      a : Fin (HAdd.hAdd n✝ 1)
      ⊢ And (LE.le (↑a) x) (LE.le x (HSub.hSub (HAdd.hAdd n✝ 1) 1)) → Exists fun a_2 …
    -/
  · exact fun hx => ⟨⟨x, Nat.lt_succ_iff.2 hx.2⟩, hx.1, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem map_valEmbedding_Ioi : (Ioi a).map Fin.valEmbedding = Ioc ↑a (n - 1) := by
  /-
    n : Nat
    a : Fin n
    ⊢ Eq (Finset.map Fin.valEmbedding (Finset.Ioi a)) (Finset.Ioc (↑a) (HSub.hSub  …
  -/
  ext x
  /-
    case h
    n : Nat
    a : Fin n
    x : Nat
    ⊢ Iff (Membership.mem (Finset.map Fin.valEmbedding (Finset.Ioi a)) x) (Members …
  -/
  simp only [exists_prop, Embedding.coe_subtype, mem_Ioi, mem_map, mem_Ioc]
  /-
    case h
    n : Nat
    a : Fin n
    x : Nat
    ⊢ Iff (Exists fun a_1 => And (LT.lt a a_1) (Eq (Fin.valEmbedding a_1) x)) (And …
  -/
  constructor
    /-
      case h.mp
      n : Nat
      a : Fin n
      x : Nat
      ⊢ (Exists fun a_1 => And (LT.lt a a_1) (Eq (Fin.valEmbedding a_1) x)) → And (L …
    -/
  · rintro ⟨x, hx, rfl⟩
    /-
      case h.mp.intro.intro
      n : Nat
      a x : Fin n
      hx : LT.lt a x
      ⊢ And (LT.lt (↑a) (Fin.valEmbedding x)) (LE.le (Fin.valEmbedding x) (HSub.hSub …
    -/
    exact ⟨hx, Nat.le_sub_of_add_le <| x.2⟩
    /-
      🎉 no goals
    -/
  /-
    case h.mpr
    n : Nat
    a : Fin n
    x : Nat
    ⊢ And (LT.lt (↑a) x) (LE.le x (HSub.hSub n 1)) → Exists fun a_2 => And (LT.lt  …
  -/
  cases n
    /-
      case h.mpr.zero
      x : Nat
      a : Fin 0
      ⊢ And (LT.lt (↑a) x) (LE.le x (HSub.hSub 0 1)) → Exists fun a_2 => And (LT.lt  …
    -/
  · exact Fin.elim0 a
    /-
      🎉 no goals
    -/
    /-
      case h.mpr.succ
      x n✝ : Nat
      a : Fin (HAdd.hAdd n✝ 1)
      ⊢ And (LT.lt (↑a) x) (LE.le x (HSub.hSub (HAdd.hAdd n✝ 1) 1)) → Exists fun a_2 …
    -/
  · exact fun hx => ⟨⟨x, Nat.lt_succ_iff.2 hx.2⟩, hx.1, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem map_valEmbedding_Iic : (Iic b).map Fin.valEmbedding = Iic ↑b := by
  /-
    n : Nat
    b : Fin n
    ⊢ Eq (Finset.map Fin.valEmbedding (Finset.Iic b)) (Finset.Iic ↑b)
  -/
  simp [Iic_eq_finset_subtype, Finset.fin, Finset.map_map, Iic_filter_lt_of_lt_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_valEmbedding_Iio : (Iio b).map Fin.valEmbedding = Iio ↑b := by
  /-
    n : Nat
    b : Fin n
    ⊢ Eq (Finset.map Fin.valEmbedding (Finset.Iio b)) (Finset.Iio ↑b)
  -/
  simp [Iio_eq_finset_subtype, Finset.fin, Finset.map_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_Ici : #(Ici a) = n - a := by
  cases n with
  | zero => exact Fin.elim0 a
  | succ =>
    rw [← card_map, map_valEmbedding_Ici, Nat.card_Icc, Nat.add_one_sub_one]


@[simp]
                                              /-
                                                n : Nat
                                                a : Fin n
                                                ⊢ Eq (Finset.Ioi a).card (HSub.hSub (HSub.hSub n 1) ↑a)
                                              -/
theorem card_Ioi : #(Ioi a) = n - 1 - a := by rw [← card_map, map_valEmbedding_Ioi, Nat.card_Ioc]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
                                          /-
                                            n : Nat
                                            b : Fin n
                                            ⊢ Eq (Finset.Iic b).card (HAdd.hAdd (↑b) 1)
                                          -/
theorem card_Iic : #(Iic b) = b + 1 := by rw [← Nat.card_Iic b, ← map_valEmbedding_Iic, card_map]
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
                                      /-
                                        n : Nat
                                        b : Fin n
                                        ⊢ Eq (Finset.Iio b).card ↑b
                                      -/
theorem card_Iio : #(Iio b) = b := by rw [← Nat.card_Iio b, ← map_valEmbedding_Iio, card_map]
                                      /-
                                        🎉 no goals
                                      -/


theorem card_fintypeIci : Fintype.card (Set.Ici a) = n - a := by
  /-
    n : Nat
    a : Fin n
    ⊢ Eq (Fintype.card ↑(Set.Ici a)) (HSub.hSub n ↑a)
  -/
  rw [Fintype.card_ofFinset, card_Ici]
  /-
    🎉 no goals
  -/


theorem card_fintypeIoi : Fintype.card (Set.Ioi a) = n - 1 - a := by
  /-
    n : Nat
    a : Fin n
    ⊢ Eq (Fintype.card ↑(Set.Ioi a)) (HSub.hSub (HSub.hSub n 1) ↑a)
  -/
  rw [Fintype.card_ofFinset, card_Ioi]
  /-
    🎉 no goals
  -/


theorem card_fintypeIic : Fintype.card (Set.Iic b) = b + 1 := by
  /-
    n : Nat
    b : Fin n
    ⊢ Eq (Fintype.card ↑(Set.Iic b)) (HAdd.hAdd (↑b) 1)
  -/
  rw [Fintype.card_ofFinset, card_Iic]
  /-
    🎉 no goals
  -/


theorem card_fintypeIio : Fintype.card (Set.Iio b) = b := by
  /-
    n : Nat
    b : Fin n
    ⊢ Eq (Fintype.card ↑(Set.Iio b)) ↑b
  -/
  rw [Fintype.card_ofFinset, card_Iio]
  /-
    🎉 no goals
  -/


