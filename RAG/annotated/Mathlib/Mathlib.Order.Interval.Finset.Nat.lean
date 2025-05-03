instance instLocallyFiniteOrder : LocallyFiniteOrder ℕ where
                                               /-
                                                 a✝ b✝ c a b : Nat
                                                 ⊢ LT.lt 0 1
                                               -/
  finsetIcc a b := ⟨List.range' a (b + 1 - a), List.nodup_range' _ _⟩
                                               /-
                                                 🎉 no goals
                                               -/
                                           /-
                                             a✝ b✝ c a b : Nat
                                             ⊢ LT.lt 0 1
                                           -/
  finsetIco a b := ⟨List.range' a (b - a), List.nodup_range' _ _⟩
                                           /-
                                             🎉 no goals
                                           -/
                                                 /-
                                                   a✝ b✝ c a b : Nat
                                                   ⊢ LT.lt 0 1
                                                 -/
  finsetIoc a b := ⟨List.range' (a + 1) (b - a), List.nodup_range' _ _⟩
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                     /-
                                                       a✝ b✝ c a b : Nat
                                                       ⊢ LT.lt 0 1
                                                     -/
  finsetIoo a b := ⟨List.range' (a + 1) (b - a - 1), List.nodup_range' _ _⟩
                                                     /-
                                                       🎉 no goals
                                                     -/
                             /-
                               a✝ b✝ c a b x : Nat
                               ⊢ Iff (Membership.mem ((fun a b => { val := ↑(List.range' a (HSub.hSub (HAdd.h …
                             -/
  finset_mem_Icc a b x := by rw [Finset.mem_mk, Multiset.mem_coe, List.mem_range'_1]; omega
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
                             /-
                               a✝ b✝ c a b x : Nat
                               ⊢ Iff (Membership.mem ((fun a b => { val := ↑(List.range' a (HSub.hSub b a)),  …
                             -/
  finset_mem_Ico a b x := by rw [Finset.mem_mk, Multiset.mem_coe, List.mem_range'_1]; omega
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
                             /-
                               a✝ b✝ c a b x : Nat
                               ⊢ Iff (Membership.mem ((fun a b => { val := ↑(List.range' (HAdd.hAdd a 1) (HSu …
                             -/
  finset_mem_Ioc a b x := by rw [Finset.mem_mk, Multiset.mem_coe, List.mem_range'_1]; omega
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
                             /-
                               a✝ b✝ c a b x : Nat
                               ⊢ Iff (Membership.mem ((fun a b => { val := ↑(List.range' (HAdd.hAdd a 1) (HSu …
                             -/
  finset_mem_Ioo a b x := by rw [Finset.mem_mk, Multiset.mem_coe, List.mem_range'_1]; omega
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


                                                              /-
                                                                a b c : Nat
                                                                ⊢ LT.lt 0 1
                                                              -/
theorem Icc_eq_range' : Icc a b = ⟨List.range' a (b + 1 - a), List.nodup_range' _ _⟩ :=
                                                              /-
                                                                🎉 no goals
                                                              -/
  rfl


                                                          /-
                                                            a b c : Nat
                                                            ⊢ LT.lt 0 1
                                                          -/
theorem Ico_eq_range' : Ico a b = ⟨List.range' a (b - a), List.nodup_range' _ _⟩ :=
                                                          /-
                                                            🎉 no goals
                                                          -/
  rfl


                                                                /-
                                                                  a b c : Nat
                                                                  ⊢ LT.lt 0 1
                                                                -/
theorem Ioc_eq_range' : Ioc a b = ⟨List.range' (a + 1) (b - a), List.nodup_range' _ _⟩ :=
                                                                /-
                                                                  🎉 no goals
                                                                -/
  rfl


                                                                    /-
                                                                      a b c : Nat
                                                                      ⊢ LT.lt 0 1
                                                                    -/
theorem Ioo_eq_range' : Ioo a b = ⟨List.range' (a + 1) (b - a - 1), List.nodup_range' _ _⟩ :=
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  rfl


theorem uIcc_eq_range' :
                                                               /-
                                                                 a b c : Nat
                                                                 ⊢ LT.lt 0 1
                                                               -/
    uIcc a b = ⟨List.range' (min a b) (max a b + 1 - min a b), List.nodup_range' _ _⟩ := rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem Iio_eq_range : Iio = range := by
  /-
    ⊢ Eq Finset.Iio Finset.range
  -/
  ext b x
  /-
    case h.h
    b x : Nat
    ⊢ Iff (Membership.mem (Finset.Iio b) x) (Membership.mem (Finset.range b) x)
  -/
  rw [mem_Iio, mem_range]
  /-
    🎉 no goals
  -/


@[simp]
                                                /-
                                                  ⊢ Eq (Finset.Ico 0) Finset.range
                                                -/
theorem Ico_zero_eq_range : Ico 0 = range := by rw [← Nat.bot_eq_zero, ← Iio_eq_Ico, Iio_eq_range]
                                                /-
                                                  🎉 no goals
                                                -/


lemma range_eq_Icc_zero_sub_one (n : ℕ) (hn : n ≠ 0) : range n = Icc 0 (n - 1) := by
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Eq (Finset.range n) (Finset.Icc 0 (HSub.hSub n 1))
  -/
  ext b
  /-
    case h
    n : Nat
    hn : Ne n 0
    b : Nat
    ⊢ Iff (Membership.mem (Finset.range n) b) (Membership.mem (Finset.Icc 0 (HSub. …
  -/
  simp_all only [mem_Icc, zero_le, true_and, mem_range]
  /-
    case h
    n : Nat
    hn : Ne n 0
    b : Nat
    ⊢ Iff (LT.lt b n) (LE.le b (HSub.hSub n 1))
  -/
  exact lt_iff_le_pred (zero_lt_of_ne_zero hn)
  /-
    🎉 no goals
  -/


theorem _root_.Finset.range_eq_Ico : range = Ico 0 :=
  Ico_zero_eq_range.symm


@[simp] lemma card_Icc : #(Icc a b) = b + 1 - a := List.length_range' ..

@[simp] lemma card_Ico : #(Ico a b) = b - a := List.length_range' ..

@[simp] lemma card_Ioc : #(Ioc a b) = b - a := List.length_range' ..

@[simp] lemma card_Ioo : #(Ioo a b) = b - a - 1 := List.length_range' ..


@[simp]
theorem card_uIcc : #(uIcc a b) = (b - a : ℤ).natAbs + 1 :=
                             /-
                               a b : Nat
                               ⊢ Eq (HSub.hSub (HAdd.hAdd (Max.max a b) 1) (Min.min a b)) (HAdd.hAdd (HSub.hS …
                             -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  (card_Icc _ _).trans <| by rw [← Int.natCast_inj, Int.ofNat_sub] <;> omega
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
                                        /-
                                          b : Nat
                                          ⊢ Eq (Finset.Iic b).card (HAdd.hAdd b 1)
                                        -/
lemma card_Iic : #(Iic b) = b + 1 := by rw [Iic_eq_Icc, card_Icc, Nat.bot_eq_zero, Nat.sub_zero]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
                                      /-
                                        b : Nat
                                        ⊢ Eq (Finset.Iio b).card b
                                      -/
theorem card_Iio : #(Iio b) = b := by rw [Iio_eq_Ico, card_Ico, Nat.bot_eq_zero, Nat.sub_zero]
                                      /-
                                        🎉 no goals
                                      -/


theorem card_fintypeIcc : Fintype.card (Set.Icc a b) = b + 1 - a := by
  /-
    a b : Nat
    ⊢ Eq (Fintype.card ↑(Set.Icc a b)) (HSub.hSub (HAdd.hAdd b 1) a)
  -/
  rw [Fintype.card_ofFinset, card_Icc]
  /-
    🎉 no goals
  -/


theorem card_fintypeIco : Fintype.card (Set.Ico a b) = b - a := by
  /-
    a b : Nat
    ⊢ Eq (Fintype.card ↑(Set.Ico a b)) (HSub.hSub b a)
  -/
  rw [Fintype.card_ofFinset, card_Ico]
  /-
    🎉 no goals
  -/


theorem card_fintypeIoc : Fintype.card (Set.Ioc a b) = b - a := by
  /-
    a b : Nat
    ⊢ Eq (Fintype.card ↑(Set.Ioc a b)) (HSub.hSub b a)
  -/
  rw [Fintype.card_ofFinset, card_Ioc]
  /-
    🎉 no goals
  -/


theorem card_fintypeIoo : Fintype.card (Set.Ioo a b) = b - a - 1 := by
  /-
    a b : Nat
    ⊢ Eq (Fintype.card ↑(Set.Ioo a b)) (HSub.hSub (HSub.hSub b a) 1)
  -/
  rw [Fintype.card_ofFinset, card_Ioo]
  /-
    🎉 no goals
  -/


theorem card_fintypeIic : Fintype.card (Set.Iic b) = b + 1 := by
  /-
    b : Nat
    ⊢ Eq (Fintype.card ↑(Set.Iic b)) (HAdd.hAdd b 1)
  -/
  rw [Fintype.card_ofFinset, card_Iic]
  /-
    🎉 no goals
  -/


                                                             /-
                                                               b : Nat
                                                               ⊢ Eq (Fintype.card ↑(Set.Iio b)) b
                                                             -/
theorem card_fintypeIio : Fintype.card (Set.Iio b) = b := by rw [Fintype.card_ofFinset, card_Iio]
                                                             /-
                                                               🎉 no goals
                                                             -/

-- TODO@Yaël: Generalize all the following lemmas to `SuccOrder`

theorem Icc_succ_left : Icc a.succ b = Ioc a b := by
  /-
    a b : Nat
    ⊢ Eq (Finset.Icc a.succ b) (Finset.Ioc a b)
  -/
  ext x
  /-
    case h
    a b x : Nat
    ⊢ Iff (Membership.mem (Finset.Icc a.succ b) x) (Membership.mem (Finset.Ioc a b …
  -/
  rw [mem_Icc, mem_Ioc, succ_le_iff]
  /-
    🎉 no goals
  -/


theorem Ico_succ_right : Ico a b.succ = Icc a b := by
  /-
    a b : Nat
    ⊢ Eq (Finset.Ico a b.succ) (Finset.Icc a b)
  -/
  ext x
  /-
    case h
    a b x : Nat
    ⊢ Iff (Membership.mem (Finset.Ico a b.succ) x) (Membership.mem (Finset.Icc a b …
  -/
  rw [mem_Ico, mem_Icc, Nat.lt_succ_iff]
  /-
    🎉 no goals
  -/


theorem Ico_succ_left : Ico a.succ b = Ioo a b := by
  /-
    a b : Nat
    ⊢ Eq (Finset.Ico a.succ b) (Finset.Ioo a b)
  -/
  ext x
  /-
    case h
    a b x : Nat
    ⊢ Iff (Membership.mem (Finset.Ico a.succ b) x) (Membership.mem (Finset.Ioo a b …
  -/
  rw [mem_Ico, mem_Ioo, succ_le_iff]
  /-
    🎉 no goals
  -/


theorem Icc_pred_right {b : ℕ} (h : 0 < b) : Icc a (b - 1) = Ico a b := by
  /-
    a b : Nat
    h : LT.lt 0 b
    ⊢ Eq (Finset.Icc a (HSub.hSub b 1)) (Finset.Ico a b)
  -/
  ext x
  /-
    case h
    a b : Nat
    h : LT.lt 0 b
    x : Nat
    ⊢ Iff (Membership.mem (Finset.Icc a (HSub.hSub b 1)) x) (Membership.mem (Finse …
  -/
  rw [mem_Icc, mem_Ico, lt_iff_le_pred h]
  /-
    🎉 no goals
  -/


theorem Ico_succ_succ : Ico a.succ b.succ = Ioc a b := by
  /-
    a b : Nat
    ⊢ Eq (Finset.Ico a.succ b.succ) (Finset.Ioc a b)
  -/
  ext x
  /-
    case h
    a b x : Nat
    ⊢ Iff (Membership.mem (Finset.Ico a.succ b.succ) x) (Membership.mem (Finset.Io …
  -/
  rw [mem_Ico, mem_Ioc, succ_le_iff, Nat.lt_succ_iff]
  /-
    🎉 no goals
  -/


@[simp]
                                                       /-
                                                         a : Nat
                                                         ⊢ Eq (Finset.Ico a (HAdd.hAdd a 1)) (Singleton.singleton a)
                                                       -/
theorem Ico_succ_singleton : Ico a (a + 1) = {a} := by rw [Ico_succ_right, Icc_self]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem Ico_pred_singleton {a : ℕ} (h : 0 < a) : Ico (a - 1) a = {a - 1} := by
  /-
    a : Nat
    h : LT.lt 0 a
    ⊢ Eq (Finset.Ico (HSub.hSub a 1) a) (Singleton.singleton (HSub.hSub a 1))
  -/
  rw [← Icc_pred_right _ h, Icc_self]
  /-
    🎉 no goals
  -/


@[simp]
                                                           /-
                                                             b : Nat
                                                             ⊢ Eq (Finset.Ioc b (HAdd.hAdd b 1)) (Singleton.singleton (HAdd.hAdd b 1))
                                                           -/
theorem Ioc_succ_singleton : Ioc b (b + 1) = {b + 1} := by rw [← Nat.Icc_succ_left, Icc_self]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem Ico_succ_right_eq_insert_Ico (h : a ≤ b) : Ico a (b + 1) = insert b (Ico a b) := by
  /-
    a b : Nat
    h : LE.le a b
    ⊢ Eq (Finset.Ico a (HAdd.hAdd b 1)) (Insert.insert b (Finset.Ico a b))
  -/
  rw [Ico_succ_right, ← Ico_insert_right h]
  /-
    🎉 no goals
  -/


theorem Ico_insert_succ_left (h : a < b) : insert a (Ico a.succ b) = Ico a b := by
  /-
    a b : Nat
    h : LT.lt a b
    ⊢ Eq (Insert.insert a (Finset.Ico a.succ b)) (Finset.Ico a b)
  -/
  rw [Ico_succ_left, ← Ioo_insert_left h]
  /-
    🎉 no goals
  -/


lemma Icc_insert_succ_left (h : a ≤ b) : insert a (Icc (a + 1) b) = Icc a b := by
  /-
    a b : Nat
    h : LE.le a b
    ⊢ Eq (Insert.insert a (Finset.Icc (HAdd.hAdd a 1) b)) (Finset.Icc a b)
  -/
  ext x
  /-
    case h
    a b : Nat
    h : LE.le a b
    x : Nat
    ⊢ Iff (Membership.mem (Insert.insert a (Finset.Icc (HAdd.hAdd a 1) b)) x) (Mem …
  -/
  simp only [mem_insert, mem_Icc]
  /-
    case h
    a b : Nat
    h : LE.le a b
    x : Nat
    ⊢ Iff (Or (Eq x a) (And (LE.le (HAdd.hAdd a 1) x) (LE.le x b))) (And (LE.le a  …
  -/
  omega
  /-
    🎉 no goals
  -/


lemma Icc_insert_succ_right (h : a ≤ b + 1) : insert (b + 1) (Icc a b) = Icc a (b + 1) := by
  /-
    a b : Nat
    h : LE.le a (HAdd.hAdd b 1)
    ⊢ Eq (Insert.insert (HAdd.hAdd b 1) (Finset.Icc a b)) (Finset.Icc a (HAdd.hAdd …
  -/
  ext x
  /-
    case h
    a b : Nat
    h : LE.le a (HAdd.hAdd b 1)
    x : Nat
    ⊢ Iff (Membership.mem (Insert.insert (HAdd.hAdd b 1) (Finset.Icc a b)) x) (Mem …
  -/
  simp only [mem_insert, mem_Icc]
  /-
    case h
    a b : Nat
    h : LE.le a (HAdd.hAdd b 1)
    x : Nat
    ⊢ Iff (Or (Eq x (HAdd.hAdd b 1)) (And (LE.le a x) (LE.le x b))) (And (LE.le a  …
  -/
  omega
  /-
    🎉 no goals
  -/


theorem image_sub_const_Ico (h : c ≤ a) :
    ((Ico a b).image fun x => x - c) = Ico (a - c) (b - c) := by
  /-
    a b c : Nat
    h : LE.le c a
    ⊢ Eq (Finset.image (fun x => HSub.hSub x c) (Finset.Ico a b)) (Finset.Ico (HSu …
  -/
  ext x
  /-
    case h
    a b c : Nat
    h : LE.le c a
    x : Nat
    ⊢ Iff (Membership.mem (Finset.image (fun x => HSub.hSub x c) (Finset.Ico a b)) …
  -/
  simp_rw [mem_image, mem_Ico]
  /-
    case h
    a b c : Nat
    h : LE.le c a
    x : Nat
    ⊢ Iff (Exists fun a_1 => And (And (LE.le a a_1) (LT.lt a_1 b)) (Eq (HSub.hSub  …
  -/
  refine ⟨?_, fun h ↦ ⟨x + c, by omega⟩⟩
  /-
    case h
    a b c : Nat
    h : LE.le c a
    x : Nat
    ⊢ (Exists fun a_1 => And (And (LE.le a a_1) (LT.lt a_1 b)) (Eq (HSub.hSub a_1  …
  -/
  rintro ⟨x, hx, rfl⟩
  /-
    case h.intro.intro
    a b c : Nat
    h : LE.le c a
    x : Nat
    hx : And (LE.le a x) (LT.lt x b)
    ⊢ And (LE.le (HSub.hSub a c) (HSub.hSub x c)) (LT.lt (HSub.hSub x c) (HSub.hSu …
  -/
  omega
  /-
    🎉 no goals
  -/


theorem Ico_image_const_sub_eq_Ico (hac : a ≤ c) :
    ((Ico a b).image fun x => c - x) = Ico (c + 1 - b) (c + 1 - a) := by
  /-
    a b c : Nat
    hac : LE.le a c
    ⊢ Eq (Finset.image (fun x => HSub.hSub c x) (Finset.Ico a b)) (Finset.Ico (HSu …
  -/
  ext x
  /-
    case h
    a b c : Nat
    hac : LE.le a c
    x : Nat
    ⊢ Iff (Membership.mem (Finset.image (fun x => HSub.hSub c x) (Finset.Ico a b)) …
  -/
  simp_rw [mem_image, mem_Ico]
  /-
    case h
    a b c : Nat
    hac : LE.le a c
    x : Nat
    ⊢ Iff (Exists fun a_1 => And (And (LE.le a a_1) (LT.lt a_1 b)) (Eq (HSub.hSub  …
  -/
  refine ⟨?_, fun h ↦ ⟨c - x, by omega⟩⟩
  /-
    case h
    a b c : Nat
    hac : LE.le a c
    x : Nat
    ⊢ (Exists fun a_1 => And (And (LE.le a a_1) (LT.lt a_1 b)) (Eq (HSub.hSub c a_ …
  -/
  rintro ⟨x, hx, rfl⟩
  /-
    case h.intro.intro
    a b c : Nat
    hac : LE.le a c
    x : Nat
    hx : And (LE.le a x) (LT.lt x b)
    ⊢ And (LE.le (HSub.hSub (HAdd.hAdd c 1) b) (HSub.hSub c x)) (LT.lt (HSub.hSub  …
  -/
  omega
  /-
    🎉 no goals
  -/


theorem Ico_succ_left_eq_erase_Ico : Ico a.succ b = erase (Ico a b) a := by
  /-
    a b : Nat
    ⊢ Eq (Finset.Ico a.succ b) ((Finset.Ico a b).erase a)
  -/
  ext x
  rw [Ico_succ_left, mem_erase, mem_Ico, mem_Ioo, ← and_assoc, ne_comm,
    and_comm (a := a ≠ x), lt_iff_le_and_ne]


theorem mod_injOn_Ico (n a : ℕ) : Set.InjOn (· % a) (Finset.Ico n (n + a)) := by
  /-
    n a : Nat
    ⊢ Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.Ico n (HAdd.hAdd n a))
  -/
  induction' n with n ih
    /-
      case zero
      a : Nat
      ⊢ Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.Ico 0 (HAdd.hAdd 0 a))
    -/
  · simp only [zero_add, Ico_zero_eq_range]
    /-
      case zero
      a : Nat
      ⊢ Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.range a)
    -/
    rintro k hk l hl (hkl : k % a = l % a)
    /-
      case zero
      a k : Nat
      hk : Membership.mem (↑(Finset.range a)) k
      l : Nat
      hl : Membership.mem (↑(Finset.range a)) l
      hkl : Eq (HMod.hMod k a) (HMod.hMod l a)
      ⊢ Eq k l
    -/
    simp only [Finset.mem_range, Finset.mem_coe] at hk hl
    /-
      case zero
      a k l : Nat
      hkl : Eq (HMod.hMod k a) (HMod.hMod l a)
      hk : LT.lt k a
      hl : LT.lt l a
      ⊢ Eq k l
    -/
    rwa [mod_eq_of_lt hk, mod_eq_of_lt hl] at hkl
    /-
      🎉 no goals
    -/
  rw [Ico_succ_left_eq_erase_Ico, succ_add, succ_eq_add_one,
    Ico_succ_right_eq_insert_Ico (by omega)]
  /-
    case succ
    a n : Nat
    ih : Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.Ico n (HAdd.hAdd n a))
    ⊢ Set.InjOn (fun x => HMod.hMod x a) ↑((Insert.insert (HAdd.hAdd n a) (Finset. …
  -/
  rintro k hk l hl (hkl : k % a = l % a)
  /-
    case succ
    a n : Nat
    ih : Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.Ico n (HAdd.hAdd n a))
    k : Nat
    hk : Membership.mem (↑((Insert.insert (HAdd.hAdd n a) (Finset.Ico n (HAdd.hAdd …
    l : Nat
    hl : Membership.mem (↑((Insert.insert (HAdd.hAdd n a) (Finset.Ico n (HAdd.hAdd …
    hkl : Eq (HMod.hMod k a) (HMod.hMod l a)
    ⊢ Eq k l
  -/
  have ha : 0 < a := Nat.pos_iff_ne_zero.2 <| by rintro rfl; simp at hk
  /-
    case succ
    a n : Nat
    ih : Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.Ico n (HAdd.hAdd n a))
    k : Nat
    hk : Membership.mem (↑((Insert.insert (HAdd.hAdd n a) (Finset.Ico n (HAdd.hAdd …
    l : Nat
    hl : Membership.mem (↑((Insert.insert (HAdd.hAdd n a) (Finset.Ico n (HAdd.hAdd …
    hkl : Eq (HMod.hMod k a) (HMod.hMod l a)
    ha : LT.lt 0 a
    ⊢ Eq k l
  -/
  simp only [Finset.mem_coe, Finset.mem_insert, Finset.mem_erase] at hk hl
  /-
    case succ
    a n : Nat
    ih : Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.Ico n (HAdd.hAdd n a))
    k l : Nat
    hkl : Eq (HMod.hMod k a) (HMod.hMod l a)
    ha : LT.lt 0 a
    hk : And (Ne k n) (Or (Eq k (HAdd.hAdd n a)) (Membership.mem (Finset.Ico n (HA …
    hl : And (Ne l n) (Or (Eq l (HAdd.hAdd n a)) (Membership.mem (Finset.Ico n (HA …
    ⊢ Eq k l
  -/
  rcases hk with ⟨hkn, rfl | hk⟩ <;> rcases hl with ⟨hln, rfl | hl⟩
    /-
      case succ.intro.inl.intro.inl
      a n : Nat
      ih : Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.Ico n (HAdd.hAdd n a))
      ha : LT.lt 0 a
      hkn : Ne (HAdd.hAdd n a) n
      hkl : Eq (HMod.hMod (HAdd.hAdd n a) a) (HMod.hMod (HAdd.hAdd n a) a)
      hln : Ne (HAdd.hAdd n a) n
      ⊢ Eq (HAdd.hAdd n a) (HAdd.hAdd n a)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case succ.intro.inl.intro.inr
      a n : Nat
      ih : Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.Ico n (HAdd.hAdd n a))
      l : Nat
      ha : LT.lt 0 a
      hkl : Eq (HMod.hMod (HAdd.hAdd n a) a) (HMod.hMod l a)
      hkn : Ne (HAdd.hAdd n a) n
      hln : Ne l n
      hl : Membership.mem (Finset.Ico n (HAdd.hAdd n a)) l
      ⊢ Eq (HAdd.hAdd n a) l
    -/
  · rw [add_mod_right] at hkl
    /-
      case succ.intro.inl.intro.inr
      a n : Nat
      ih : Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.Ico n (HAdd.hAdd n a))
      l : Nat
      ha : LT.lt 0 a
      hkl : Eq (HMod.hMod n a) (HMod.hMod l a)
      hkn : Ne (HAdd.hAdd n a) n
      hln : Ne l n
      hl : Membership.mem (Finset.Ico n (HAdd.hAdd n a)) l
      ⊢ Eq (HAdd.hAdd n a) l
    -/
    refine (hln <| ih hl ?_ hkl.symm).elim
    /-
      case succ.intro.inl.intro.inr
      a n : Nat
      ih : Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.Ico n (HAdd.hAdd n a))
      l : Nat
      ha : LT.lt 0 a
      hkl : Eq (HMod.hMod n a) (HMod.hMod l a)
      hkn : Ne (HAdd.hAdd n a) n
      hln : Ne l n
      hl : Membership.mem (Finset.Ico n (HAdd.hAdd n a)) l
      ⊢ Membership.mem (↑(Finset.Ico n (HAdd.hAdd n a))) n
    -/
    simpa using Nat.lt_add_of_pos_right (n := n) ha
    /-
      🎉 no goals
    -/
    /-
      case succ.intro.inr.intro.inl
      a n : Nat
      ih : Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.Ico n (HAdd.hAdd n a))
      k : Nat
      ha : LT.lt 0 a
      hkn : Ne k n
      hk : Membership.mem (Finset.Ico n (HAdd.hAdd n a)) k
      hkl : Eq (HMod.hMod k a) (HMod.hMod (HAdd.hAdd n a) a)
      hln : Ne (HAdd.hAdd n a) n
      ⊢ Eq k (HAdd.hAdd n a)
    -/
  · rw [add_mod_right] at hkl
    /-
      case succ.intro.inr.intro.inl
      a n : Nat
      ih : Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.Ico n (HAdd.hAdd n a))
      k : Nat
      ha : LT.lt 0 a
      hkn : Ne k n
      hk : Membership.mem (Finset.Ico n (HAdd.hAdd n a)) k
      hkl : Eq (HMod.hMod k a) (HMod.hMod n a)
      hln : Ne (HAdd.hAdd n a) n
      ⊢ Eq k (HAdd.hAdd n a)
    -/
    suffices k = n by contradiction
    /-
      case succ.intro.inr.intro.inl
      a n : Nat
      ih : Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.Ico n (HAdd.hAdd n a))
      k : Nat
      ha : LT.lt 0 a
      hkn : Ne k n
      hk : Membership.mem (Finset.Ico n (HAdd.hAdd n a)) k
      hkl : Eq (HMod.hMod k a) (HMod.hMod n a)
      hln : Ne (HAdd.hAdd n a) n
      ⊢ Eq k n
    -/
    refine ih hk ?_ hkl
    /-
      case succ.intro.inr.intro.inl
      a n : Nat
      ih : Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.Ico n (HAdd.hAdd n a))
      k : Nat
      ha : LT.lt 0 a
      hkn : Ne k n
      hk : Membership.mem (Finset.Ico n (HAdd.hAdd n a)) k
      hkl : Eq (HMod.hMod k a) (HMod.hMod n a)
      hln : Ne (HAdd.hAdd n a) n
      ⊢ Membership.mem (↑(Finset.Ico n (HAdd.hAdd n a))) n
    -/
    simpa using Nat.lt_add_of_pos_right (n := n) ha
    /-
      🎉 no goals
    -/
    /-
      case succ.intro.inr.intro.inr
      a n : Nat
      ih : Set.InjOn (fun x => HMod.hMod x a) ↑(Finset.Ico n (HAdd.hAdd n a))
      k l : Nat
      hkl : Eq (HMod.hMod k a) (HMod.hMod l a)
      ha : LT.lt 0 a
      hkn : Ne k n
      hk : Membership.mem (Finset.Ico n (HAdd.hAdd n a)) k
      hln : Ne l n
      hl : Membership.mem (Finset.Ico n (HAdd.hAdd n a)) l
      ⊢ Eq k l
    -/
                            /-
                              🎉 no goals
                            -/
  · refine ih ?_ ?_ hkl <;> simp only [Finset.mem_coe, hk, hl]
                            /-
                              🎉 no goals
                            -/


/-- Note that while this lemma cannot be easily generalized to a type class, it holds for ℤ as
well. See `Int.image_Ico_emod` for the ℤ version. -/
theorem image_Ico_mod (n a : ℕ) : (Ico n (n + a)).image (· % a) = range a := by
  /-
    n a : Nat
    ⊢ Eq (Finset.image (fun x => HMod.hMod x a) (Finset.Ico n (HAdd.hAdd n a))) (F …
  -/
  obtain rfl | ha := eq_or_ne a 0
    /-
      case inl
      n : Nat
      ⊢ Eq (Finset.image (fun x => HMod.hMod x 0) (Finset.Ico n (HAdd.hAdd n 0))) (F …
    -/
  · rw [range_zero, add_zero, Ico_self, image_empty]
    /-
      🎉 no goals
    -/
  /-
    case inr
    n a : Nat
    ha : Ne a 0
    ⊢ Eq (Finset.image (fun x => HMod.hMod x a) (Finset.Ico n (HAdd.hAdd n a))) (F …
  -/
  ext i
  /-
    case inr.h
    n a : Nat
    ha : Ne a 0
    i : Nat
    ⊢ Iff (Membership.mem (Finset.image (fun x => HMod.hMod x a) (Finset.Ico n (HA …
  -/
  simp only [mem_image, exists_prop, mem_range, mem_Ico]
  /-
    case inr.h
    n a : Nat
    ha : Ne a 0
    i : Nat
    ⊢ Iff (Exists fun a_1 => And (And (LE.le n a_1) (LT.lt a_1 (HAdd.hAdd n a))) ( …
  -/
  constructor
    /-
      case inr.h.mp
      n a : Nat
      ha : Ne a 0
      i : Nat
      ⊢ (Exists fun a_1 => And (And (LE.le n a_1) (LT.lt a_1 (HAdd.hAdd n a))) (Eq ( …
    -/
  · rintro ⟨i, _, rfl⟩
    /-
      case inr.h.mp.intro.intro
      n a : Nat
      ha : Ne a 0
      i : Nat
      left✝ : And (LE.le n i) (LT.lt i (HAdd.hAdd n a))
      ⊢ LT.lt (HMod.hMod i a) a
    -/
    exact mod_lt i ha.bot_lt
    /-
      🎉 no goals
    -/
  /-
    case inr.h.mpr
    n a : Nat
    ha : Ne a 0
    i : Nat
    ⊢ LT.lt i a → Exists fun a_2 => And (And (LE.le n a_2) (LT.lt a_2 (HAdd.hAdd n …
  -/
  intro hia
  /-
    case inr.h.mpr
    n a : Nat
    ha : Ne a 0
    i : Nat
    hia : LT.lt i a
    ⊢ Exists fun a_1 => And (And (LE.le n a_1) (LT.lt a_1 (HAdd.hAdd n a))) (Eq (H …
  -/
  have hn := Nat.mod_add_div n a
  /-
    case inr.h.mpr
    n a : Nat
    ha : Ne a 0
    i : Nat
    hia : LT.lt i a
    hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
    ⊢ Exists fun a_1 => And (And (LE.le n a_1) (LT.lt a_1 (HAdd.hAdd n a))) (Eq (H …
  -/
  obtain hi | hi := lt_or_le i (n % a)
    /-
      case inr.h.mpr.inl
      n a : Nat
      ha : Ne a 0
      i : Nat
      hia : LT.lt i a
      hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
      hi : LT.lt i (HMod.hMod n a)
      ⊢ Exists fun a_1 => And (And (LE.le n a_1) (LT.lt a_1 (HAdd.hAdd n a))) (Eq (H …
    -/
  · refine ⟨i + a * (n / a + 1), ⟨?_, ?_⟩, ?_⟩
      /-
        case inr.h.mpr.inl.refine_1
        n a : Nat
        ha : Ne a 0
        i : Nat
        hia : LT.lt i a
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LT.lt i (HMod.hMod n a)
        ⊢ LE.le n (HAdd.hAdd i (HMul.hMul a (HAdd.hAdd (HDiv.hDiv n a) 1)))
      -/
    · rw [add_comm (n / a), Nat.mul_add, mul_one, ← add_assoc]
      /-
        case inr.h.mpr.inl.refine_1
        n a : Nat
        ha : Ne a 0
        i : Nat
        hia : LT.lt i a
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LT.lt i (HMod.hMod n a)
        ⊢ LE.le n (HAdd.hAdd (HAdd.hAdd i a) (HMul.hMul a (HDiv.hDiv n a)))
      -/
      refine hn.symm.le.trans (Nat.add_le_add_right ?_ _)
      /-
        case inr.h.mpr.inl.refine_1
        n a : Nat
        ha : Ne a 0
        i : Nat
        hia : LT.lt i a
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LT.lt i (HMod.hMod n a)
        ⊢ LE.le (HMod.hMod n a) (HAdd.hAdd i a)
      -/
      simpa only [zero_add] using add_le_add (zero_le i) (Nat.mod_lt n ha.bot_lt).le
      /-
        🎉 no goals
      -/
      /-
        case inr.h.mpr.inl.refine_2
        n a : Nat
        ha : Ne a 0
        i : Nat
        hia : LT.lt i a
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LT.lt i (HMod.hMod n a)
        ⊢ LT.lt (HAdd.hAdd i (HMul.hMul a (HAdd.hAdd (HDiv.hDiv n a) 1))) (HAdd.hAdd n …
      -/
    · refine lt_of_lt_of_le (Nat.add_lt_add_right hi (a * (n / a + 1))) ?_
      /-
        case inr.h.mpr.inl.refine_2
        n a : Nat
        ha : Ne a 0
        i : Nat
        hia : LT.lt i a
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LT.lt i (HMod.hMod n a)
        ⊢ LE.le (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HAdd.hAdd (HDiv.hDiv n a) 1)) …
      -/
      rw [Nat.mul_add, mul_one, ← add_assoc, hn]
      /-
        🎉 no goals
      -/
      /-
        case inr.h.mpr.inl.refine_3
        n a : Nat
        ha : Ne a 0
        i : Nat
        hia : LT.lt i a
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LT.lt i (HMod.hMod n a)
        ⊢ Eq (HMod.hMod (HAdd.hAdd i (HMul.hMul a (HAdd.hAdd (HDiv.hDiv n a) 1))) a) i
      -/
    · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hia]
      /-
        🎉 no goals
      -/
    /-
      case inr.h.mpr.inr
      n a : Nat
      ha : Ne a 0
      i : Nat
      hia : LT.lt i a
      hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
      hi : LE.le (HMod.hMod n a) i
      ⊢ Exists fun a_1 => And (And (LE.le n a_1) (LT.lt a_1 (HAdd.hAdd n a))) (Eq (H …
    -/
  · refine ⟨i + a * (n / a), ⟨?_, ?_⟩, ?_⟩
      /-
        case inr.h.mpr.inr.refine_1
        n a : Nat
        ha : Ne a 0
        i : Nat
        hia : LT.lt i a
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LE.le (HMod.hMod n a) i
        ⊢ LE.le n (HAdd.hAdd i (HMul.hMul a (HDiv.hDiv n a)))
      -/
    · omega
      /-
        🎉 no goals
      -/
      /-
        case inr.h.mpr.inr.refine_2
        n a : Nat
        ha : Ne a 0
        i : Nat
        hia : LT.lt i a
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LE.le (HMod.hMod n a) i
        ⊢ LT.lt (HAdd.hAdd i (HMul.hMul a (HDiv.hDiv n a))) (HAdd.hAdd n a)
      -/
    · omega
      /-
        🎉 no goals
      -/
      /-
        case inr.h.mpr.inr.refine_3
        n a : Nat
        ha : Ne a 0
        i : Nat
        hia : LT.lt i a
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LE.le (HMod.hMod n a) i
        ⊢ Eq (HMod.hMod (HAdd.hAdd i (HMul.hMul a (HDiv.hDiv n a))) a) i
      -/
    · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hia]
      /-
        🎉 no goals
      -/


theorem multiset_Ico_map_mod (n a : ℕ) :
    (Multiset.Ico n (n + a)).map (· % a) = Multiset.range a := by
  /-
    n a : Nat
    ⊢ Eq (Multiset.map (fun x => HMod.hMod x a) (Multiset.Ico n (HAdd.hAdd n a)))  …
  -/
  convert congr_arg Finset.val (image_Ico_mod n a)
  /-
    case h.e'_2
    n a : Nat
    ⊢ Eq (Multiset.map (fun x => HMod.hMod x a) (Multiset.Ico n (HAdd.hAdd n a)))  …
  -/
  refine ((nodup_map_iff_inj_on (Finset.Ico _ _).nodup).2 <| ?_).dedup.symm
  /-
    case h.e'_2
    n a : Nat
    ⊢ ∀ (x : Nat), Membership.mem (Finset.Ico n (HAdd.hAdd n a)).val x → ∀ (y : Na …
  -/
  exact mod_injOn_Ico _ _
  /-
    🎉 no goals
  -/


theorem range_image_pred_top_sub (n : ℕ) :
    ((Finset.range n).image fun j => n - 1 - j) = Finset.range n := by
  /-
    n : Nat
    ⊢ Eq (Finset.image (fun j => HSub.hSub (HSub.hSub n 1) j) (Finset.range n)) (F …
  -/
  cases n
    /-
      case zero
      ⊢ Eq (Finset.image (fun j => HSub.hSub (HSub.hSub 0 1) j) (Finset.range 0)) (F …
    -/
  · rw [range_zero, image_empty]
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      ⊢ Eq (Finset.image (fun j => HSub.hSub (HSub.hSub (HAdd.hAdd n✝ 1) 1) j) (Fins …
    -/
  · rw [Finset.range_eq_Ico, Nat.Ico_image_const_sub_eq_Ico (Nat.zero_le _)]
    /-
      case succ
      n✝ : Nat
      ⊢ Eq (Finset.Ico (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd n✝ 1) 1) 1) (HAdd …
    -/
    simp_rw [succ_sub_succ, Nat.sub_zero, Nat.sub_self]
    /-
      🎉 no goals
    -/


theorem range_add_eq_union : range (a + b) = range a ∪ (range b).map (addLeftEmbedding a) := by
  /-
    a b : Nat
    ⊢ Eq (Finset.range (HAdd.hAdd a b)) (Union.union (Finset.range a) (Finset.map  …
  -/
  rw [Finset.range_eq_Ico, map_eq_image]
  /-
    a b : Nat
    ⊢ Eq (Finset.Ico 0 (HAdd.hAdd a b)) (Union.union (Finset.Ico 0 a) (Finset.imag …
  -/
  convert (Ico_union_Ico_eq_Ico a.zero_le (a.le_add_right b)).symm
  /-
    case h.e'_3.h.e'_4
    a b : Nat
    ⊢ Eq (Finset.image (⇑(addLeftEmbedding a)) (Finset.Ico 0 b)) (Finset.Ico a (HA …
  -/
  ext x
  /-
    case h.e'_3.h.e'_4.h
    a b x : Nat
    ⊢ Iff (Membership.mem (Finset.image (⇑(addLeftEmbedding a)) (Finset.Ico 0 b))  …
  -/
  simp only [Ico_zero_eq_range, mem_image, mem_range, addLeftEmbedding_apply, mem_Ico]
  /-
    case h.e'_3.h.e'_4.h
    a b x : Nat
    ⊢ Iff (Exists fun a_1 => And (LT.lt a_1 b) (Eq (HAdd.hAdd a a_1) x)) (And (LE. …
  -/
  constructor
    /-
      case h.e'_3.h.e'_4.h.mp
      a b x : Nat
      ⊢ (Exists fun a_1 => And (LT.lt a_1 b) (Eq (HAdd.hAdd a a_1) x)) → And (LE.le  …
    -/
  · aesop
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_4.h.mpr
      a b x : Nat
      ⊢ And (LE.le a x) (LT.lt x (HAdd.hAdd a b)) → Exists fun a_2 => And (LT.lt a_2 …
    -/
  · rintro h
    /-
      case h.e'_3.h.e'_4.h.mpr
      a b x : Nat
      h : And (LE.le a x) (LT.lt x (HAdd.hAdd a b))
      ⊢ Exists fun a_1 => And (LT.lt a_1 b) (Eq (HAdd.hAdd a a_1) x)
    -/
    exact ⟨x - a, by omega⟩
    /-
      🎉 no goals
    -/


theorem Nat.decreasing_induction_of_not_bddAbove (h : ∀ n, P (n + 1) → P n)
    (hP : ¬BddAbove { x | P x }) (n : ℕ) : P n :=
  let ⟨_, hm, hl⟩ := not_bddAbove_iff.1 hP n
  decreasingInduction (fun _ _ => h _) hm hl.le


@[elab_as_elim]
lemma Nat.strong_decreasing_induction (base : ∃ n, ∀ m > n, P m) (step : ∀ n, (∀ m > n, P m) → P n)
    (n : ℕ) : P n := by
  /-
    P : Nat → Prop
    base : Exists fun n => ∀ (m : Nat), GT.gt m n → P m
    step : ∀ (n : Nat), (∀ (m : Nat), GT.gt m n → P m) → P n
    n : Nat
    ⊢ P n
  -/
  apply Nat.decreasing_induction_of_not_bddAbove (P := fun n ↦ ∀ m ≥ n, P m) _ _ n n le_rfl
    /-
      P : Nat → Prop
      base : Exists fun n => ∀ (m : Nat), GT.gt m n → P m
      step : ∀ (n : Nat), (∀ (m : Nat), GT.gt m n → P m) → P n
      n : Nat
      ⊢ ∀ (n : Nat), (fun n => ∀ (m : Nat), GE.ge m n → P m) (HAdd.hAdd n 1) → (fun  …
    -/
  · intro n ih m hm
    /-
      P : Nat → Prop
      base : Exists fun n => ∀ (m : Nat), GT.gt m n → P m
      step : ∀ (n : Nat), (∀ (m : Nat), GT.gt m n → P m) → P n
      n✝ n : Nat
      ih : ∀ (m : Nat), GE.ge m (HAdd.hAdd n 1) → P m
      m : Nat
      hm : GE.ge m n
      ⊢ P m
    -/
    rcases hm.eq_or_lt with rfl | hm
      /-
        case inl
        P : Nat → Prop
        base : Exists fun n => ∀ (m : Nat), GT.gt m n → P m
        step : ∀ (n : Nat), (∀ (m : Nat), GT.gt m n → P m) → P n
        n✝ n : Nat
        ih : ∀ (m : Nat), GE.ge m (HAdd.hAdd n 1) → P m
        hm : GE.ge n n
        ⊢ P n
      -/
    · exact step n ih
      /-
        🎉 no goals
      -/
      /-
        case inr
        P : Nat → Prop
        base : Exists fun n => ∀ (m : Nat), GT.gt m n → P m
        step : ∀ (n : Nat), (∀ (m : Nat), GT.gt m n → P m) → P n
        n✝ n : Nat
        ih : ∀ (m : Nat), GE.ge m (HAdd.hAdd n 1) → P m
        m : Nat
        hm✝ : GE.ge m n
        hm : LT.lt n m
        ⊢ P m
      -/
    · exact ih m hm
      /-
        🎉 no goals
      -/
    /-
      P : Nat → Prop
      base : Exists fun n => ∀ (m : Nat), GT.gt m n → P m
      step : ∀ (n : Nat), (∀ (m : Nat), GT.gt m n → P m) → P n
      n : Nat
      ⊢ Not (BddAbove (setOf fun x => (fun n => ∀ (m : Nat), GE.ge m n → P m) x))
    -/
  · rintro ⟨b, hb⟩
    /-
      case intro
      P : Nat → Prop
      base : Exists fun n => ∀ (m : Nat), GT.gt m n → P m
      step : ∀ (n : Nat), (∀ (m : Nat), GT.gt m n → P m) → P n
      n b : Nat
      hb : Membership.mem (upperBounds (setOf fun x => (fun n => ∀ (m : Nat), GE.ge  …
      ⊢ False
    -/
    rcases base with ⟨n, hn⟩
    /-
      case intro.intro
      P : Nat → Prop
      step : ∀ (n : Nat), (∀ (m : Nat), GT.gt m n → P m) → P n
      n✝ b : Nat
      hb : Membership.mem (upperBounds (setOf fun x => (fun n => ∀ (m : Nat), GE.ge  …
      n : Nat
      hn : ∀ (m : Nat), GT.gt m n → P m
      ⊢ False
    -/
    specialize @hb (n + b + 1) (fun m hm ↦ hn _ _)
    /-
      case intro.intro
      P : Nat → Prop
      step : ∀ (n : Nat), (∀ (m : Nat), GT.gt m n → P m) → P n
      n✝ b : Nat
      hb : Membership.mem (upperBounds (setOf fun x => (fun n => ∀ (m : Nat), GE.ge  …
      n : Nat
      hn : ∀ (m : Nat), GT.gt m n → P m
      m : Nat
      hm : GE.ge m (HAdd.hAdd (HAdd.hAdd n b) 1)
      ⊢ GT.gt m n
    -/
    all_goals omega
    /-
      🎉 no goals
    -/


theorem Nat.decreasing_induction_of_infinite
    (h : ∀ n, P (n + 1) → P n) (hP : { x | P x }.Infinite) (n : ℕ) : P n :=
  Nat.decreasing_induction_of_not_bddAbove h (mt BddAbove.finite hP) n


theorem Nat.cauchy_induction' (seed : ℕ) (h : ∀ n, P (n + 1) → P n) (hs : P seed)
    (hi : ∀ x, seed ≤ x → P x → ∃ y, x < y ∧ P y) (n : ℕ) : P n := by
  /-
    P : Nat → Prop
    seed : Nat
    h : ∀ (n : Nat), P (HAdd.hAdd n 1) → P n
    hs : P seed
    hi : ∀ (x : Nat), LE.le seed x → P x → Exists fun y => And (LT.lt x y) (P y)
    n : Nat
    ⊢ P n
  -/
  apply Nat.decreasing_induction_of_infinite h fun hf => _
  /-
    P : Nat → Prop
    seed : Nat
    h : ∀ (n : Nat), P (HAdd.hAdd n 1) → P n
    hs : P seed
    hi : ∀ (x : Nat), LE.le seed x → P x → Exists fun y => And (LT.lt x y) (P y)
    n : Nat
    ⊢ (setOf fun x => P x).Finite → False
  -/
  intro hf
  /-
    P : Nat → Prop
    seed : Nat
    h : ∀ (n : Nat), P (HAdd.hAdd n 1) → P n
    hs : P seed
    hi : ∀ (x : Nat), LE.le seed x → P x → Exists fun y => And (LT.lt x y) (P y)
    n : Nat
    hf : (setOf fun x => P x).Finite
    ⊢ False
  -/
  obtain ⟨m, hP, hm⟩ := hf.exists_maximal_wrt id _ ⟨seed, hs⟩
  /-
    case intro.intro
    P : Nat → Prop
    seed : Nat
    h : ∀ (n : Nat), P (HAdd.hAdd n 1) → P n
    hs : P seed
    hi : ∀ (x : Nat), LE.le seed x → P x → Exists fun y => And (LT.lt x y) (P y)
    n : Nat
    hf : (setOf fun x => P x).Finite
    m : Nat
    hP : Membership.mem (setOf fun x => P x) m
    hm : ∀ (a' : Nat), Membership.mem (setOf fun x => P x) a' → LE.le (id m) (id a …
    ⊢ False
  -/
  obtain ⟨y, hl, hy⟩ := hi m (le_of_not_lt fun hl => hl.ne <| hm seed hs hl.le) hP
  /-
    case intro.intro.intro.intro
    P : Nat → Prop
    seed : Nat
    h : ∀ (n : Nat), P (HAdd.hAdd n 1) → P n
    hs : P seed
    hi : ∀ (x : Nat), LE.le seed x → P x → Exists fun y => And (LT.lt x y) (P y)
    n : Nat
    hf : (setOf fun x => P x).Finite
    m : Nat
    hP : Membership.mem (setOf fun x => P x) m
    hm : ∀ (a' : Nat), Membership.mem (setOf fun x => P x) a' → LE.le (id m) (id a …
    y : Nat
    hl : LT.lt m y
    hy : P y
    ⊢ False
  -/
  exact hl.ne (hm y hy hl.le)
  /-
    🎉 no goals
  -/


theorem Nat.cauchy_induction (h : ∀ n, P (n + 1) → P n) (seed : ℕ) (hs : P seed) (f : ℕ → ℕ)
    (hf : ∀ x, seed ≤ x → P x → x < f x ∧ P (f x)) (n : ℕ) : P n :=
  seed.cauchy_induction' h hs (fun x hl hx => ⟨f x, hf x hl hx⟩) n


theorem Nat.cauchy_induction_mul (h : ∀ (n : ℕ), P (n + 1) → P n) (k seed : ℕ) (hk : 1 < k)
    (hs : P seed.succ) (hm : ∀ x, seed < x → P x → P (k * x)) (n : ℕ) : P n := by
  /-
    P : Nat → Prop
    h : ∀ (n : Nat), P (HAdd.hAdd n 1) → P n
    k seed : Nat
    hk : LT.lt 1 k
    hs : P seed.succ
    hm : ∀ (x : Nat), LT.lt seed x → P x → P (HMul.hMul k x)
    n : Nat
    ⊢ P n
  -/
  apply Nat.cauchy_induction h _ hs (k * ·) fun x hl hP => ⟨_, hm x hl hP⟩
  /-
    P : Nat → Prop
    h : ∀ (n : Nat), P (HAdd.hAdd n 1) → P n
    k seed : Nat
    hk : LT.lt 1 k
    hs : P seed.succ
    hm : ∀ (x : Nat), LT.lt seed x → P x → P (HMul.hMul k x)
    n : Nat
    ⊢ ∀ (x : Nat), LE.le seed.succ x → P x → LT.lt x ((fun x => HMul.hMul k x) x)
  -/
  intro _ hl _
  /-
    P : Nat → Prop
    h : ∀ (n : Nat), P (HAdd.hAdd n 1) → P n
    k seed : Nat
    hk : LT.lt 1 k
    hs : P seed.succ
    hm : ∀ (x : Nat), LT.lt seed x → P x → P (HMul.hMul k x)
    n x✝ : Nat
    hl : LE.le seed.succ x✝
    hP✝ : P x✝
    ⊢ LT.lt x✝ ((fun x => HMul.hMul k x) x✝)
  -/
  convert (Nat.mul_lt_mul_right <| seed.succ_pos.trans_le hl).2 hk
  /-
    case h.e'_3
    P : Nat → Prop
    h : ∀ (n : Nat), P (HAdd.hAdd n 1) → P n
    k seed : Nat
    hk : LT.lt 1 k
    hs : P seed.succ
    hm : ∀ (x : Nat), LT.lt seed x → P x → P (HMul.hMul k x)
    n x✝ : Nat
    hl : LE.le seed.succ x✝
    hP✝ : P x✝
    ⊢ Eq x✝ (HMul.hMul 1 x✝)
  -/
  rw [one_mul]
  /-
    🎉 no goals
  -/


theorem Nat.cauchy_induction_two_mul (h : ∀ n, P (n + 1) → P n) (seed : ℕ) (hs : P seed.succ)
    (hm : ∀ x, seed < x → P x → P (2 * x)) (n : ℕ) : P n :=
  Nat.cauchy_induction_mul h 2 seed Nat.one_lt_two hs hm n


theorem Nat.pow_imp_self_of_one_lt {M} [Monoid M] (k : ℕ) (hk : 1 < k)
    (P : M → Prop) (hmul : ∀ x y, P x → P (x * y) ∨ P (y * x))
    (hpow : ∀ x, P (x ^ k) → P x) : ∀ n x, P (x ^ n) → P x :=
  k.cauchy_induction_mul (fun n ih x hx ↦ ih x <| (hmul _ x hx).elim
                /-
                  M : Type u_1
                  inst✝ : Monoid M
                  k : Nat
                  hk : LT.lt 1 k
                  P : M → Prop
                  hmul : ∀ (x y : M), P x → Or (P (HMul.hMul x y)) (P (HMul.hMul y x))
                  hpow : ∀ (x : M), P (HPow.hPow x k) → P x
                  n : Nat
                  ih : ∀ (x : M), P (HPow.hPow x (HAdd.hAdd n 1)) → P x
                  x : M
                  hx : P (HPow.hPow x n)
                  h : P (HMul.hMul (HPow.hPow x n) x)
                  ⊢ P (HPow.hPow x (HAdd.hAdd n 1))
                -/
                /-
                  🎉 no goals
                -/
    (fun h ↦ by rwa [_root_.pow_succ]) fun h ↦ by rwa [_root_.pow_succ']) 0 hk
                                                  /-
                                                    🎉 no goals
                                                  -/
    (fun x hx ↦ pow_one x ▸ hx) fun n _ hn x hx ↦ hpow x <| hn _ <| (pow_mul x k n).subst hx


