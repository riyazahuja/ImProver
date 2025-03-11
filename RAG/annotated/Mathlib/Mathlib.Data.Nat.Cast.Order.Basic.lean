@[mono]
theorem mono_cast : Monotone (Nat.cast : ℕ → α) :=
  monotone_nat_of_le_succ fun n ↦ by
    /-
      α : Type u_1
      inst✝³ : AddMonoidWithOne α
      inst✝² : PartialOrder α
      inst✝¹ : AddLeftMono α
      inst✝ : ZeroLEOneClass α
      n : Nat
      ⊢ LE.le ↑n ↑(HAdd.hAdd n 1)
    -/
    rw [Nat.cast_succ]; exact le_add_of_nonneg_right zero_le_one
                        /-
                          🎉 no goals
                        -/


@[deprecated mono_cast (since := "2024-02-10")]
theorem cast_le_cast {a b : ℕ} (h : a ≤ b) : (a : α) ≤ b := mono_cast h


@[gcongr]
theorem _root_.GCongr.natCast_le_natCast {a b : ℕ} (h : a ≤ b) : (a : α) ≤ b := mono_cast h


/-- See also `Nat.cast_nonneg`, specialised for an `OrderedSemiring`. -/
@[simp low]
theorem cast_nonneg' (n : ℕ) : 0 ≤ (n : α) :=
  @Nat.cast_zero α _ ▸ mono_cast (Nat.zero_le n)


/-- See also `Nat.ofNat_nonneg`, specialised for an `OrderedSemiring`. -/
-- See note [no_index around OfNat.ofNat]
@[simp low]
theorem ofNat_nonneg' (n : ℕ) [n.AtLeastTwo] : 0 ≤ (no_index (OfNat.ofNat n : α)) := cast_nonneg' n


theorem cast_add_one_pos (n : ℕ) : 0 < (n : α) + 1 := by
  /-
    α : Type u_1
    inst✝⁴ : AddMonoidWithOne α
    inst✝³ : PartialOrder α
    inst✝² : AddLeftMono α
    inst✝¹ : ZeroLEOneClass α
    inst✝ : NeZero 1
    n : Nat
    ⊢ LT.lt 0 (HAdd.hAdd (↑n) 1)
  -/
  apply zero_lt_one.trans_le
  /-
    α : Type u_1
    inst✝⁴ : AddMonoidWithOne α
    inst✝³ : PartialOrder α
    inst✝² : AddLeftMono α
    inst✝¹ : ZeroLEOneClass α
    inst✝ : NeZero 1
    n : Nat
    ⊢ LE.le 1 (HAdd.hAdd (↑n) 1)
  -/
  convert (@mono_cast α _).imp (?_ : 1 ≤ n + 1)
      /-
        case h.e'_3
        α : Type u_1
        inst✝⁴ : AddMonoidWithOne α
        inst✝³ : PartialOrder α
        inst✝² : AddLeftMono α
        inst✝¹ : ZeroLEOneClass α
        inst✝ : NeZero 1
        n : Nat
        ⊢ Eq 1 ↑1
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
  <;> simp
      /-
        🎉 no goals
      -/


/-- See also `Nat.cast_pos`, specialised for an `OrderedSemiring`. -/
@[simp low]
                                                      /-
                                                        α : Type u_1
                                                        inst✝⁴ : AddMonoidWithOne α
                                                        inst✝³ : PartialOrder α
                                                        inst✝² : AddLeftMono α
                                                        inst✝¹ : ZeroLEOneClass α
                                                        inst✝ : NeZero 1
                                                        n : Nat
                                                        ⊢ Iff (LT.lt 0 ↑n) (LT.lt 0 n)
                                                      -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
theorem cast_pos' {n : ℕ} : (0 : α) < n ↔ 0 < n := by cases n <;> simp [cast_add_one_pos]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem strictMono_cast : StrictMono (Nat.cast : ℕ → α) :=
  mono_cast.strictMono_of_injective cast_injective


@[gcongr]
lemma _root_.GCongr.natCast_lt_natCast {a b : ℕ} (h : a < b) : (a : α) < b := strictMono_cast h


/-- `Nat.cast : ℕ → α` as an `OrderEmbedding` -/
@[simps! (config := .asFn)]
def castOrderEmbedding : ℕ ↪o α :=
  OrderEmbedding.ofStrictMono Nat.cast Nat.strictMono_cast


@[simp, norm_cast]
theorem cast_le : (m : α) ≤ n ↔ m ≤ n :=
  strictMono_cast.le_iff_le


@[simp, norm_cast, mono]
theorem cast_lt : (m : α) < n ↔ m < n :=
  strictMono_cast.lt_iff_lt


@[simp, norm_cast]
                                                /-
                                                  α : Type u_1
                                                  inst✝⁴ : AddMonoidWithOne α
                                                  inst✝³ : PartialOrder α
                                                  inst✝² : AddLeftMono α
                                                  inst✝¹ : ZeroLEOneClass α
                                                  inst✝ : CharZero α
                                                  n : Nat
                                                  ⊢ Iff (LT.lt 1 ↑n) (LT.lt 1 n)
                                                -/
theorem one_lt_cast : 1 < (n : α) ↔ 1 < n := by rw [← cast_one, cast_lt]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp, norm_cast]
                                                /-
                                                  α : Type u_1
                                                  inst✝⁴ : AddMonoidWithOne α
                                                  inst✝³ : PartialOrder α
                                                  inst✝² : AddLeftMono α
                                                  inst✝¹ : ZeroLEOneClass α
                                                  inst✝ : CharZero α
                                                  n : Nat
                                                  ⊢ Iff (LE.le 1 ↑n) (LE.le 1 n)
                                                -/
theorem one_le_cast : 1 ≤ (n : α) ↔ 1 ≤ n := by rw [← cast_one, cast_le]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp, norm_cast]
theorem cast_lt_one : (n : α) < 1 ↔ n = 0 := by
  /-
    α : Type u_1
    inst✝⁴ : AddMonoidWithOne α
    inst✝³ : PartialOrder α
    inst✝² : AddLeftMono α
    inst✝¹ : ZeroLEOneClass α
    inst✝ : CharZero α
    n : Nat
    ⊢ Iff (LT.lt (↑n) 1) (Eq n 0)
  -/
  rw [← cast_one, cast_lt, Nat.lt_succ_iff, le_zero]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
                                                /-
                                                  α : Type u_1
                                                  inst✝⁴ : AddMonoidWithOne α
                                                  inst✝³ : PartialOrder α
                                                  inst✝² : AddLeftMono α
                                                  inst✝¹ : ZeroLEOneClass α
                                                  inst✝ : CharZero α
                                                  n : Nat
                                                  ⊢ Iff (LE.le (↑n) 1) (LE.le n 1)
                                                -/
theorem cast_le_one : (n : α) ≤ 1 ↔ n ≤ 1 := by rw [← cast_one, cast_le]
                                                /-
                                                  🎉 no goals
                                                -/


                                                      /-
                                                        α : Type u_1
                                                        inst✝⁴ : AddMonoidWithOne α
                                                        inst✝³ : PartialOrder α
                                                        inst✝² : AddLeftMono α
                                                        inst✝¹ : ZeroLEOneClass α
                                                        inst✝ : CharZero α
                                                        n : Nat
                                                        ⊢ Iff (LE.le (↑n) 0) (Eq n 0)
                                                      -/
@[simp] lemma cast_nonpos : (n : α) ≤ 0 ↔ n = 0 := by norm_cast; omega
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem ofNat_le_cast : (no_index (OfNat.ofNat m : α)) ≤ n ↔ (OfNat.ofNat m : ℕ) ≤ n :=
  cast_le

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem ofNat_lt_cast : (no_index (OfNat.ofNat m : α)) < n ↔ (OfNat.ofNat m : ℕ) < n :=
  cast_lt


@[simp]
theorem cast_le_ofNat : (m : α) ≤ (no_index (OfNat.ofNat n)) ↔ m ≤ OfNat.ofNat n :=
  cast_le

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem cast_lt_ofNat : (m : α) < (no_index (OfNat.ofNat n)) ↔ m < OfNat.ofNat n :=
  cast_lt

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem one_lt_ofNat : 1 < (no_index (OfNat.ofNat n : α)) :=
  one_lt_cast.mpr AtLeastTwo.one_lt

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem one_le_ofNat : 1 ≤ (no_index (OfNat.ofNat n : α)) :=
  one_le_cast.mpr NeZero.one_le

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem not_ofNat_le_one : ¬(no_index (OfNat.ofNat n : α)) ≤ 1 :=
  (cast_le_one.not.trans not_le).mpr AtLeastTwo.one_lt

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem not_ofNat_lt_one : ¬(no_index (OfNat.ofNat n : α)) < 1 :=
  mt le_of_lt not_ofNat_le_one


theorem ofNat_le :
    (no_index (OfNat.ofNat m : α)) ≤ (no_index (OfNat.ofNat n)) ↔
      (OfNat.ofNat m : ℕ) ≤ OfNat.ofNat n :=
  cast_le

-- See note [no_index around OfNat.ofNat]
-- @[simp]

theorem ofNat_lt :
    (no_index (OfNat.ofNat m : α)) < (no_index (OfNat.ofNat n)) ↔
      (OfNat.ofNat m : ℕ) < OfNat.ofNat n :=
  cast_lt


instance [AddMonoidWithOne α] [CharZero α] : Nontrivial α where exists_pair_ne :=
                                                         /-
                                                           α : Type u_1
                                                           inst✝¹ : AddMonoidWithOne α
                                                           inst✝ : CharZero α
                                                           ⊢ Ne 1 0
                                                         -/
  ⟨1, 0, (Nat.cast_one (R := α) ▸ Nat.cast_ne_zero.2 (by decide))⟩
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem NeZero.nat_of_injective {n : ℕ} [NeZero (n : R)] [RingHomClass F R S] {f : F}
    (hf : Function.Injective f) : NeZero (n : S) :=
                                             /-
                                               R : Type u_2
                                               S : Type u_3
                                               F : Type u_4
                                               inst✝⁴ : NonAssocSemiring R
                                               inst✝³ : NonAssocSemiring S
                                               inst✝² : FunLike F R S
                                               n : Nat
                                               inst✝¹ : NeZero ↑n
                                               inst✝ : RingHomClass F R S
                                               f : F
                                               hf : Function.Injective ⇑f
                                               h : Eq (↑n) 0
                                               ⊢ Eq (f ↑n) (f 0)
                                             -/
  ⟨fun h ↦ NeZero.natCast_ne n R <| hf <| by simpa only [map_natCast, map_zero f]⟩
                                             /-
                                               🎉 no goals
                                             -/


