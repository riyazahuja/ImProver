instance instLocallyFiniteOrder : LocallyFiniteOrder ℤ where
  finsetIcc a b :=
    (Finset.range (b + 1 - a).toNat).map <| Nat.castEmbedding.trans <| addLeftEmbedding a
  finsetIco a b := (Finset.range (b - a).toNat).map <| Nat.castEmbedding.trans <| addLeftEmbedding a
  finsetIoc a b :=
    (Finset.range (b - a).toNat).map <| Nat.castEmbedding.trans <| addLeftEmbedding (a + 1)
  finsetIoo a b :=
    (Finset.range (b - a - 1).toNat).map <| Nat.castEmbedding.trans <| addLeftEmbedding (a + 1)
  finset_mem_Icc a b x := by
    simp_rw [mem_map, mem_range, Int.lt_toNat, Function.Embedding.trans_apply,
      Nat.castEmbedding_apply, addLeftEmbedding_apply]
    /-
      a b x : Int
      ⊢ Iff (Exists fun a_1 => And (LT.lt (↑a_1) (HSub.hSub (HAdd.hAdd b 1) a)) (Eq  …
    -/
    constructor
      /-
        case mp
        a b x : Int
        ⊢ (Exists fun a_1 => And (LT.lt (↑a_1) (HSub.hSub (HAdd.hAdd b 1) a)) (Eq (HAd …
      -/
    · rintro ⟨a, h, rfl⟩
      /-
        case mp.intro.intro
        a✝ b : Int
        a : Nat
        h : LT.lt (↑a) (HSub.hSub (HAdd.hAdd b 1) a✝)
        ⊢ And (LE.le a✝ (HAdd.hAdd a✝ ↑a)) (LE.le (HAdd.hAdd a✝ ↑a) b)
      -/
      rw [lt_sub_iff_add_lt, Int.lt_add_one_iff, add_comm] at h
      /-
        case mp.intro.intro
        a✝ b : Int
        a : Nat
        h : LE.le (HAdd.hAdd a✝ ↑a) b
        ⊢ And (LE.le a✝ (HAdd.hAdd a✝ ↑a)) (LE.le (HAdd.hAdd a✝ ↑a) b)
      -/
      exact ⟨Int.le.intro a rfl, h⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr
        a b x : Int
        ⊢ And (LE.le a x) (LE.le x b) → Exists fun a_2 => And (LT.lt (↑a_2) (HSub.hSub …
      -/
    · rintro ⟨ha, hb⟩
      /-
        case mpr.intro
        a b x : Int
        ha : LE.le a x
        hb : LE.le x b
        ⊢ Exists fun a_1 => And (LT.lt (↑a_1) (HSub.hSub (HAdd.hAdd b 1) a)) (Eq (HAdd …
      -/
      use (x - a).toNat
      /-
        case h
        a b x : Int
        ha : LE.le a x
        hb : LE.le x b
        ⊢ And (LT.lt (↑(HSub.hSub x a).toNat) (HSub.hSub (HAdd.hAdd b 1) a)) (Eq (HAdd …
      -/
      rw [← lt_add_one_iff] at hb
      /-
        case h
        a b x : Int
        ha : LE.le a x
        hb : LT.lt x (HAdd.hAdd b 1)
        ⊢ And (LT.lt (↑(HSub.hSub x a).toNat) (HSub.hSub (HAdd.hAdd b 1) a)) (Eq (HAdd …
      -/
      rw [toNat_sub_of_le ha]
      /-
        case h
        a b x : Int
        ha : LE.le a x
        hb : LT.lt x (HAdd.hAdd b 1)
        ⊢ And (LT.lt (HSub.hSub x a) (HSub.hSub (HAdd.hAdd b 1) a)) (Eq (HAdd.hAdd a ( …
      -/
      exact ⟨sub_lt_sub_right hb _, add_sub_cancel _ _⟩
      /-
        🎉 no goals
      -/
  finset_mem_Ico a b x := by
    simp_rw [mem_map, mem_range, Int.lt_toNat, Function.Embedding.trans_apply,
      Nat.castEmbedding_apply, addLeftEmbedding_apply]
    /-
      a b x : Int
      ⊢ Iff (Exists fun a_1 => And (LT.lt (↑a_1) (HSub.hSub b a)) (Eq (HAdd.hAdd a ↑ …
    -/
    constructor
      /-
        case mp
        a b x : Int
        ⊢ (Exists fun a_1 => And (LT.lt (↑a_1) (HSub.hSub b a)) (Eq (HAdd.hAdd a ↑a_1) …
      -/
    · rintro ⟨a, h, rfl⟩
      /-
        case mp.intro.intro
        a✝ b : Int
        a : Nat
        h : LT.lt (↑a) (HSub.hSub b a✝)
        ⊢ And (LE.le a✝ (HAdd.hAdd a✝ ↑a)) (LT.lt (HAdd.hAdd a✝ ↑a) b)
      -/
      exact ⟨Int.le.intro a rfl, lt_sub_iff_add_lt'.mp h⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr
        a b x : Int
        ⊢ And (LE.le a x) (LT.lt x b) → Exists fun a_2 => And (LT.lt (↑a_2) (HSub.hSub …
      -/
    · rintro ⟨ha, hb⟩
      /-
        case mpr.intro
        a b x : Int
        ha : LE.le a x
        hb : LT.lt x b
        ⊢ Exists fun a_1 => And (LT.lt (↑a_1) (HSub.hSub b a)) (Eq (HAdd.hAdd a ↑a_1) x)
      -/
      use (x - a).toNat
      /-
        case h
        a b x : Int
        ha : LE.le a x
        hb : LT.lt x b
        ⊢ And (LT.lt (↑(HSub.hSub x a).toNat) (HSub.hSub b a)) (Eq (HAdd.hAdd a ↑(HSub …
      -/
      rw [toNat_sub_of_le ha]
      /-
        case h
        a b x : Int
        ha : LE.le a x
        hb : LT.lt x b
        ⊢ And (LT.lt (HSub.hSub x a) (HSub.hSub b a)) (Eq (HAdd.hAdd a (HSub.hSub x a) …
      -/
      exact ⟨sub_lt_sub_right hb _, add_sub_cancel _ _⟩
      /-
        🎉 no goals
      -/
  finset_mem_Ioc a b x := by
    simp_rw [mem_map, mem_range, Int.lt_toNat, Function.Embedding.trans_apply,
      Nat.castEmbedding_apply, addLeftEmbedding_apply]
    /-
      a b x : Int
      ⊢ Iff (Exists fun a_1 => And (LT.lt (↑a_1) (HSub.hSub b a)) (Eq (HAdd.hAdd (HA …
    -/
    constructor
      /-
        case mp
        a b x : Int
        ⊢ (Exists fun a_1 => And (LT.lt (↑a_1) (HSub.hSub b a)) (Eq (HAdd.hAdd (HAdd.h …
      -/
    · rintro ⟨a, h, rfl⟩
      /-
        case mp.intro.intro
        a✝ b : Int
        a : Nat
        h : LT.lt (↑a) (HSub.hSub b a✝)
        ⊢ And (LT.lt a✝ (HAdd.hAdd (HAdd.hAdd a✝ 1) ↑a)) (LE.le (HAdd.hAdd (HAdd.hAdd  …
      -/
      rw [← add_one_le_iff, le_sub_iff_add_le', add_comm _ (1 : ℤ), ← add_assoc] at h
      /-
        case mp.intro.intro
        a✝ b : Int
        a : Nat
        h : LE.le (HAdd.hAdd (HAdd.hAdd a✝ 1) ↑a) b
        ⊢ And (LT.lt a✝ (HAdd.hAdd (HAdd.hAdd a✝ 1) ↑a)) (LE.le (HAdd.hAdd (HAdd.hAdd  …
      -/
      exact ⟨Int.le.intro a rfl, h⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr
        a b x : Int
        ⊢ And (LT.lt a x) (LE.le x b) → Exists fun a_2 => And (LT.lt (↑a_2) (HSub.hSub …
      -/
    · rintro ⟨ha, hb⟩
      /-
        case mpr.intro
        a b x : Int
        ha : LT.lt a x
        hb : LE.le x b
        ⊢ Exists fun a_1 => And (LT.lt (↑a_1) (HSub.hSub b a)) (Eq (HAdd.hAdd (HAdd.hA …
      -/
      use (x - (a + 1)).toNat
      /-
        case h
        a b x : Int
        ha : LT.lt a x
        hb : LE.le x b
        ⊢ And (LT.lt (↑(HSub.hSub x (HAdd.hAdd a 1)).toNat) (HSub.hSub b a)) (Eq (HAdd …
      -/
      rw [toNat_sub_of_le ha, ← add_one_le_iff, sub_add, add_sub_cancel_right]
      /-
        case h
        a b x : Int
        ha : LT.lt a x
        hb : LE.le x b
        ⊢ And (LE.le (HSub.hSub x a) (HSub.hSub b a)) (Eq (HAdd.hAdd (HAdd.hAdd a 1) ( …
      -/
      exact ⟨sub_le_sub_right hb _, add_sub_cancel _ _⟩
      /-
        🎉 no goals
      -/
  finset_mem_Ioo a b x := by
    simp_rw [mem_map, mem_range, Int.lt_toNat, Function.Embedding.trans_apply,
      Nat.castEmbedding_apply, addLeftEmbedding_apply]
    /-
      a b x : Int
      ⊢ Iff (Exists fun a_1 => And (LT.lt (↑a_1) (HSub.hSub (HSub.hSub b a) 1)) (Eq  …
    -/
    constructor
      /-
        case mp
        a b x : Int
        ⊢ (Exists fun a_1 => And (LT.lt (↑a_1) (HSub.hSub (HSub.hSub b a) 1)) (Eq (HAd …
      -/
    · rintro ⟨a, h, rfl⟩
      /-
        case mp.intro.intro
        a✝ b : Int
        a : Nat
        h : LT.lt (↑a) (HSub.hSub (HSub.hSub b a✝) 1)
        ⊢ And (LT.lt a✝ (HAdd.hAdd (HAdd.hAdd a✝ 1) ↑a)) (LT.lt (HAdd.hAdd (HAdd.hAdd  …
      -/
      rw [sub_sub, lt_sub_iff_add_lt'] at h
      /-
        case mp.intro.intro
        a✝ b : Int
        a : Nat
        h : LT.lt (HAdd.hAdd (HAdd.hAdd a✝ 1) ↑a) b
        ⊢ And (LT.lt a✝ (HAdd.hAdd (HAdd.hAdd a✝ 1) ↑a)) (LT.lt (HAdd.hAdd (HAdd.hAdd  …
      -/
      exact ⟨Int.le.intro a rfl, h⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr
        a b x : Int
        ⊢ And (LT.lt a x) (LT.lt x b) → Exists fun a_2 => And (LT.lt (↑a_2) (HSub.hSub …
      -/
    · rintro ⟨ha, hb⟩
      /-
        case mpr.intro
        a b x : Int
        ha : LT.lt a x
        hb : LT.lt x b
        ⊢ Exists fun a_1 => And (LT.lt (↑a_1) (HSub.hSub (HSub.hSub b a) 1)) (Eq (HAdd …
      -/
      use (x - (a + 1)).toNat
      /-
        case h
        a b x : Int
        ha : LT.lt a x
        hb : LT.lt x b
        ⊢ And (LT.lt (↑(HSub.hSub x (HAdd.hAdd a 1)).toNat) (HSub.hSub (HSub.hSub b a) …
      -/
      rw [toNat_sub_of_le ha, sub_sub]
      /-
        case h
        a b x : Int
        ha : LT.lt a x
        hb : LT.lt x b
        ⊢ And (LT.lt (HSub.hSub x (HAdd.hAdd a 1)) (HSub.hSub b (HAdd.hAdd a 1))) (Eq  …
      -/
      exact ⟨sub_lt_sub_right hb _, add_sub_cancel _ _⟩
      /-
        🎉 no goals
      -/


theorem Icc_eq_finset_map :
    Icc a b =
      (Finset.range (b + 1 - a).toNat).map (Nat.castEmbedding.trans <| addLeftEmbedding a) :=
  rfl


theorem Ico_eq_finset_map :
    Ico a b = (Finset.range (b - a).toNat).map (Nat.castEmbedding.trans <| addLeftEmbedding a) :=
  rfl


theorem Ioc_eq_finset_map :
    Ioc a b =
      (Finset.range (b - a).toNat).map (Nat.castEmbedding.trans <| addLeftEmbedding (a + 1)) :=
  rfl


theorem Ioo_eq_finset_map :
    Ioo a b =
      (Finset.range (b - a - 1).toNat).map (Nat.castEmbedding.trans <| addLeftEmbedding (a + 1)) :=
  rfl


theorem uIcc_eq_finset_map :
    uIcc a b = (range (max a b + 1 - min a b).toNat).map
      (Nat.castEmbedding.trans <| addLeftEmbedding <| min a b) := rfl


@[simp]
theorem card_Icc : #(Icc a b) = (b + 1 - a).toNat := (card_map _).trans <| card_range _


@[simp]
theorem card_Ico : #(Ico a b) = (b - a).toNat := (card_map _).trans <| card_range _


@[simp]
theorem card_Ioc : #(Ioc a b) = (b - a).toNat := (card_map _).trans <| card_range _


@[simp]
theorem card_Ioo : #(Ioo a b) = (b - a - 1).toNat := (card_map _).trans <| card_range _


@[simp]
theorem card_uIcc : #(uIcc a b) = (b - a).natAbs + 1 :=
  (card_map _).trans <|
    (Nat.cast_inj (R := ℤ)).mp <| by
      rw [card_range,
        Int.toNat_of_nonneg (sub_nonneg_of_le <| le_add_one min_le_max), Int.ofNat_add,
        Int.natCast_natAbs, add_comm, add_sub_assoc, max_sub_min_eq_abs, add_comm, Int.ofNat_one]


theorem card_Icc_of_le (h : a ≤ b + 1) : (#(Icc a b) : ℤ) = b + 1 - a := by
  /-
    a b : Int
    h : LE.le a (HAdd.hAdd b 1)
    ⊢ Eq (↑(Finset.Icc a b).card) (HSub.hSub (HAdd.hAdd b 1) a)
  -/
  rw [card_Icc, toNat_sub_of_le h]
  /-
    🎉 no goals
  -/


theorem card_Ico_of_le (h : a ≤ b) : (#(Ico a b) : ℤ) = b - a := by
  /-
    a b : Int
    h : LE.le a b
    ⊢ Eq (↑(Finset.Ico a b).card) (HSub.hSub b a)
  -/
  rw [card_Ico, toNat_sub_of_le h]
  /-
    🎉 no goals
  -/


theorem card_Ioc_of_le (h : a ≤ b) : (#(Ioc a b) : ℤ) = b - a := by
  /-
    a b : Int
    h : LE.le a b
    ⊢ Eq (↑(Finset.Ioc a b).card) (HSub.hSub b a)
  -/
  rw [card_Ioc, toNat_sub_of_le h]
  /-
    🎉 no goals
  -/


theorem card_Ioo_of_lt (h : a < b) : (#(Ioo a b) : ℤ) = b - a - 1 := by
  /-
    a b : Int
    h : LT.lt a b
    ⊢ Eq (↑(Finset.Ioo a b).card) (HSub.hSub (HSub.hSub b a) 1)
  -/
  rw [card_Ioo, sub_sub, toNat_sub_of_le h]
  /-
    🎉 no goals
  -/


theorem Icc_eq_pair : Finset.Icc a (a + 1) = {a, a + 1} := by
  /-
    a : Int
    ⊢ Eq (Finset.Icc a (HAdd.hAdd a 1)) (Insert.insert a (Singleton.singleton (HAd …
  -/
  ext
  /-
    case h
    a a✝ : Int
    ⊢ Iff (Membership.mem (Finset.Icc a (HAdd.hAdd a 1)) a✝) (Membership.mem (Inse …
  -/
  simp
  /-
    case h
    a a✝ : Int
    ⊢ Iff (And (LE.le a a✝) (LE.le a✝ (HAdd.hAdd a 1))) (Or (Eq a✝ a) (Eq a✝ (HAdd …
  -/
  omega
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): removed `simp` attribute because `simpNF` says it can prove it

theorem card_fintype_Icc : Fintype.card (Set.Icc a b) = (b + 1 - a).toNat := by
  /-
    a b : Int
    ⊢ Eq (Fintype.card ↑(Set.Icc a b)) (HSub.hSub (HAdd.hAdd b 1) a).toNat
  -/
  rw [← card_Icc, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): removed `simp` attribute because `simpNF` says it can prove it

theorem card_fintype_Ico : Fintype.card (Set.Ico a b) = (b - a).toNat := by
  /-
    a b : Int
    ⊢ Eq (Fintype.card ↑(Set.Ico a b)) (HSub.hSub b a).toNat
  -/
  rw [← card_Ico, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): removed `simp` attribute because `simpNF` says it can prove it

theorem card_fintype_Ioc : Fintype.card (Set.Ioc a b) = (b - a).toNat := by
  /-
    a b : Int
    ⊢ Eq (Fintype.card ↑(Set.Ioc a b)) (HSub.hSub b a).toNat
  -/
  rw [← card_Ioc, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): removed `simp` attribute because `simpNF` says it can prove it

theorem card_fintype_Ioo : Fintype.card (Set.Ioo a b) = (b - a - 1).toNat := by
  /-
    a b : Int
    ⊢ Eq (Fintype.card ↑(Set.Ioo a b)) (HSub.hSub (HSub.hSub b a) 1).toNat
  -/
  rw [← card_Ioo, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/


theorem card_fintype_uIcc : Fintype.card (Set.uIcc a b) = (b - a).natAbs + 1 := by
  /-
    a b : Int
    ⊢ Eq (Fintype.card ↑(Set.uIcc a b)) (HAdd.hAdd (HSub.hSub b a).natAbs 1)
  -/
  rw [← card_uIcc, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/


theorem card_fintype_Icc_of_le (h : a ≤ b + 1) : (Fintype.card (Set.Icc a b) : ℤ) = b + 1 - a := by
  /-
    a b : Int
    h : LE.le a (HAdd.hAdd b 1)
    ⊢ Eq (↑(Fintype.card ↑(Set.Icc a b))) (HSub.hSub (HAdd.hAdd b 1) a)
  -/
  rw [card_fintype_Icc, toNat_sub_of_le h]
  /-
    🎉 no goals
  -/


theorem card_fintype_Ico_of_le (h : a ≤ b) : (Fintype.card (Set.Ico a b) : ℤ) = b - a := by
  /-
    a b : Int
    h : LE.le a b
    ⊢ Eq (↑(Fintype.card ↑(Set.Ico a b))) (HSub.hSub b a)
  -/
  rw [card_fintype_Ico, toNat_sub_of_le h]
  /-
    🎉 no goals
  -/


theorem card_fintype_Ioc_of_le (h : a ≤ b) : (Fintype.card (Set.Ioc a b) : ℤ) = b - a := by
  /-
    a b : Int
    h : LE.le a b
    ⊢ Eq (↑(Fintype.card ↑(Set.Ioc a b))) (HSub.hSub b a)
  -/
  rw [card_fintype_Ioc, toNat_sub_of_le h]
  /-
    🎉 no goals
  -/


theorem card_fintype_Ioo_of_lt (h : a < b) : (Fintype.card (Set.Ioo a b) : ℤ) = b - a - 1 := by
  /-
    a b : Int
    h : LT.lt a b
    ⊢ Eq (↑(Fintype.card ↑(Set.Ioo a b))) (HSub.hSub (HSub.hSub b a) 1)
  -/
  rw [card_fintype_Ioo, sub_sub, toNat_sub_of_le h]
  /-
    🎉 no goals
  -/


theorem image_Ico_emod (n a : ℤ) (h : 0 ≤ a) : (Ico n (n + a)).image (· % a) = Ico 0 a := by
  /-
    n a : Int
    h : LE.le 0 a
    ⊢ Eq (Finset.image (fun x => HMod.hMod x a) (Finset.Ico n (HAdd.hAdd n a))) (F …
  -/
  obtain rfl | ha := eq_or_lt_of_le h
    /-
      case inl
      n : Int
      h : LE.le 0 0
      ⊢ Eq (Finset.image (fun x => HMod.hMod x 0) (Finset.Ico n (HAdd.hAdd n 0))) (F …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    n a : Int
    h : LE.le 0 a
    ha : LT.lt 0 a
    ⊢ Eq (Finset.image (fun x => HMod.hMod x a) (Finset.Ico n (HAdd.hAdd n a))) (F …
  -/
  ext i
  /-
    case inr.h
    n a : Int
    h : LE.le 0 a
    ha : LT.lt 0 a
    i : Int
    ⊢ Iff (Membership.mem (Finset.image (fun x => HMod.hMod x a) (Finset.Ico n (HA …
  -/
  simp only [mem_image, mem_range, mem_Ico]
  /-
    case inr.h
    n a : Int
    h : LE.le 0 a
    ha : LT.lt 0 a
    i : Int
    ⊢ Iff (Exists fun a_1 => And (And (LE.le n a_1) (LT.lt a_1 (HAdd.hAdd n a))) ( …
  -/
  constructor
    /-
      case inr.h.mp
      n a : Int
      h : LE.le 0 a
      ha : LT.lt 0 a
      i : Int
      ⊢ (Exists fun a_1 => And (And (LE.le n a_1) (LT.lt a_1 (HAdd.hAdd n a))) (Eq ( …
    -/
  · rintro ⟨i, _, rfl⟩
    /-
      case inr.h.mp.intro.intro
      n a : Int
      h : LE.le 0 a
      ha : LT.lt 0 a
      i : Int
      left✝ : And (LE.le n i) (LT.lt i (HAdd.hAdd n a))
      ⊢ And (LE.le 0 (HMod.hMod i a)) (LT.lt (HMod.hMod i a) a)
    -/
    exact ⟨emod_nonneg i ha.ne', emod_lt_of_pos i ha⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.h.mpr
    n a : Int
    h : LE.le 0 a
    ha : LT.lt 0 a
    i : Int
    ⊢ And (LE.le 0 i) (LT.lt i a) → Exists fun a_2 => And (And (LE.le n a_2) (LT.l …
  -/
  intro hia
  /-
    case inr.h.mpr
    n a : Int
    h : LE.le 0 a
    ha : LT.lt 0 a
    i : Int
    hia : And (LE.le 0 i) (LT.lt i a)
    ⊢ Exists fun a_1 => And (And (LE.le n a_1) (LT.lt a_1 (HAdd.hAdd n a))) (Eq (H …
  -/
  have hn := Int.emod_add_ediv n a
  /-
    case inr.h.mpr
    n a : Int
    h : LE.le 0 a
    ha : LT.lt 0 a
    i : Int
    hia : And (LE.le 0 i) (LT.lt i a)
    hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
    ⊢ Exists fun a_1 => And (And (LE.le n a_1) (LT.lt a_1 (HAdd.hAdd n a))) (Eq (H …
  -/
  obtain hi | hi := lt_or_le i (n % a)
    /-
      case inr.h.mpr.inl
      n a : Int
      h : LE.le 0 a
      ha : LT.lt 0 a
      i : Int
      hia : And (LE.le 0 i) (LT.lt i a)
      hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
      hi : LT.lt i (HMod.hMod n a)
      ⊢ Exists fun a_1 => And (And (LE.le n a_1) (LT.lt a_1 (HAdd.hAdd n a))) (Eq (H …
    -/
  · refine ⟨i + a * (n / a + 1), ⟨?_, ?_⟩, ?_⟩
      /-
        case inr.h.mpr.inl.refine_1
        n a : Int
        h : LE.le 0 a
        ha : LT.lt 0 a
        i : Int
        hia : And (LE.le 0 i) (LT.lt i a)
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LT.lt i (HMod.hMod n a)
        ⊢ LE.le n (HAdd.hAdd i (HMul.hMul a (HAdd.hAdd (HDiv.hDiv n a) 1)))
      -/
    · rw [add_comm (n / a), mul_add, mul_one, ← add_assoc]
      /-
        case inr.h.mpr.inl.refine_1
        n a : Int
        h : LE.le 0 a
        ha : LT.lt 0 a
        i : Int
        hia : And (LE.le 0 i) (LT.lt i a)
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LT.lt i (HMod.hMod n a)
        ⊢ LE.le n (HAdd.hAdd (HAdd.hAdd i a) (HMul.hMul a (HDiv.hDiv n a)))
      -/
      refine hn.symm.le.trans (add_le_add_right ?_ _)
      /-
        case inr.h.mpr.inl.refine_1
        n a : Int
        h : LE.le 0 a
        ha : LT.lt 0 a
        i : Int
        hia : And (LE.le 0 i) (LT.lt i a)
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LT.lt i (HMod.hMod n a)
        ⊢ LE.le (HMod.hMod n a) (HAdd.hAdd i a)
      -/
      simpa only [zero_add] using add_le_add hia.left (Int.emod_lt_of_pos n ha).le
      /-
        🎉 no goals
      -/
      /-
        case inr.h.mpr.inl.refine_2
        n a : Int
        h : LE.le 0 a
        ha : LT.lt 0 a
        i : Int
        hia : And (LE.le 0 i) (LT.lt i a)
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LT.lt i (HMod.hMod n a)
        ⊢ LT.lt (HAdd.hAdd i (HMul.hMul a (HAdd.hAdd (HDiv.hDiv n a) 1))) (HAdd.hAdd n …
      -/
    · refine lt_of_lt_of_le (add_lt_add_right hi (a * (n / a + 1))) ?_
      /-
        case inr.h.mpr.inl.refine_2
        n a : Int
        h : LE.le 0 a
        ha : LT.lt 0 a
        i : Int
        hia : And (LE.le 0 i) (LT.lt i a)
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LT.lt i (HMod.hMod n a)
        ⊢ LE.le (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HAdd.hAdd (HDiv.hDiv n a) 1)) …
      -/
      rw [mul_add, mul_one, ← add_assoc, hn]
      /-
        🎉 no goals
      -/
      /-
        case inr.h.mpr.inl.refine_3
        n a : Int
        h : LE.le 0 a
        ha : LT.lt 0 a
        i : Int
        hia : And (LE.le 0 i) (LT.lt i a)
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LT.lt i (HMod.hMod n a)
        ⊢ Eq (HMod.hMod (HAdd.hAdd i (HMul.hMul a (HAdd.hAdd (HDiv.hDiv n a) 1))) a) i
      -/
    · rw [Int.add_mul_emod_self_left, Int.emod_eq_of_lt hia.left hia.right]
      /-
        🎉 no goals
      -/
    /-
      case inr.h.mpr.inr
      n a : Int
      h : LE.le 0 a
      ha : LT.lt 0 a
      i : Int
      hia : And (LE.le 0 i) (LT.lt i a)
      hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
      hi : LE.le (HMod.hMod n a) i
      ⊢ Exists fun a_1 => And (And (LE.le n a_1) (LT.lt a_1 (HAdd.hAdd n a))) (Eq (H …
    -/
  · refine ⟨i + a * (n / a), ⟨?_, ?_⟩, ?_⟩
      /-
        case inr.h.mpr.inr.refine_1
        n a : Int
        h : LE.le 0 a
        ha : LT.lt 0 a
        i : Int
        hia : And (LE.le 0 i) (LT.lt i a)
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LE.le (HMod.hMod n a) i
        ⊢ LE.le n (HAdd.hAdd i (HMul.hMul a (HDiv.hDiv n a)))
      -/
    · exact hn.symm.le.trans (add_le_add_right hi _)
      /-
        🎉 no goals
      -/
      /-
        case inr.h.mpr.inr.refine_2
        n a : Int
        h : LE.le 0 a
        ha : LT.lt 0 a
        i : Int
        hia : And (LE.le 0 i) (LT.lt i a)
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LE.le (HMod.hMod n a) i
        ⊢ LT.lt (HAdd.hAdd i (HMul.hMul a (HDiv.hDiv n a))) (HAdd.hAdd n a)
      -/
    · rw [add_comm n a]
      /-
        case inr.h.mpr.inr.refine_2
        n a : Int
        h : LE.le 0 a
        ha : LT.lt 0 a
        i : Int
        hia : And (LE.le 0 i) (LT.lt i a)
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LE.le (HMod.hMod n a) i
        ⊢ LT.lt (HAdd.hAdd i (HMul.hMul a (HDiv.hDiv n a))) (HAdd.hAdd a n)
      -/
      refine add_lt_add_of_lt_of_le hia.right (le_trans ?_ hn.le)
      /-
        case inr.h.mpr.inr.refine_2
        n a : Int
        h : LE.le 0 a
        ha : LT.lt 0 a
        i : Int
        hia : And (LE.le 0 i) (LT.lt i a)
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LE.le (HMod.hMod n a) i
        ⊢ LE.le (HMul.hMul a (HDiv.hDiv n a)) (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a  …
      -/
      simp only [Nat.zero_le, le_add_iff_nonneg_left]
      /-
        case inr.h.mpr.inr.refine_2
        n a : Int
        h : LE.le 0 a
        ha : LT.lt 0 a
        i : Int
        hia : And (LE.le 0 i) (LT.lt i a)
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LE.le (HMod.hMod n a) i
        ⊢ LE.le 0 (HMod.hMod n a)
      -/
      exact Int.emod_nonneg n (ne_of_gt ha)
      /-
        🎉 no goals
      -/
      /-
        case inr.h.mpr.inr.refine_3
        n a : Int
        h : LE.le 0 a
        ha : LT.lt 0 a
        i : Int
        hia : And (LE.le 0 i) (LT.lt i a)
        hn : Eq (HAdd.hAdd (HMod.hMod n a) (HMul.hMul a (HDiv.hDiv n a))) n
        hi : LE.le (HMod.hMod n a) i
        ⊢ Eq (HMod.hMod (HAdd.hAdd i (HMul.hMul a (HDiv.hDiv n a))) a) i
      -/
    · rw [Int.add_mul_emod_self_left, Int.emod_eq_of_lt hia.left hia.right]
      /-
        🎉 no goals
      -/


