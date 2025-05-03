/-- This function sends finite cardinals to the corresponding natural, and infinite cardinals
  to 0. -/
noncomputable def toNat : Cardinal →*₀ ℕ :=
  ENat.toNatHom.comp toENat


@[simp] lemma toNat_toENat (a : Cardinal) : ENat.toNat (toENat a) = toNat a := rfl


@[simp]
theorem toNat_ofENat (n : ℕ∞) : toNat n = ENat.toNat n :=
  congr_arg ENat.toNat <| toENat_ofENat n


@[simp, norm_cast] theorem toNat_natCast (n : ℕ) : toNat n = n := toNat_ofENat n


@[simp]
lemma toNat_eq_zero : toNat c = 0 ↔ c = 0 ∨ ℵ₀ ≤ c := by
  /-
    c : Cardinal.{u}
    ⊢ Iff (Eq (Cardinal.toNat c) 0) (Or (Eq c 0) (LE.le Cardinal.aleph0 c))
  -/
  rw [← toNat_toENat, ENat.toNat_eq_zero, toENat_eq_zero, toENat_eq_top]
  /-
    🎉 no goals
  -/


                                                         /-
                                                           c : Cardinal.{u}
                                                           ⊢ Iff (Ne (Cardinal.toNat c) 0) (And (Ne c 0) (LT.lt c Cardinal.aleph0))
                                                         -/
lemma toNat_ne_zero : toNat c ≠ 0 ↔ c ≠ 0 ∧ c < ℵ₀ := by simp [not_or]
                                                         /-
                                                           🎉 no goals
                                                         -/

@[simp] lemma toNat_pos : 0 < toNat c ↔ c ≠ 0 ∧ c < ℵ₀ := pos_iff_ne_zero.trans toNat_ne_zero


theorem cast_toNat_of_lt_aleph0 {c : Cardinal} (h : c < ℵ₀) : ↑(toNat c) = c := by
  /-
    c : Cardinal.{u_1}
    h : LT.lt c Cardinal.aleph0
    ⊢ Eq (↑(Cardinal.toNat c)) c
  -/
  lift c to ℕ using h
  /-
    case intro
    c : Nat
    ⊢ Eq ↑(Cardinal.toNat ↑c) ↑c
  -/
  rw [toNat_natCast]
  /-
    🎉 no goals
  -/


theorem toNat_apply_of_lt_aleph0 {c : Cardinal.{u}} (h : c < ℵ₀) :
    toNat c = Classical.choose (lt_aleph0.1 h) :=
  Nat.cast_injective (R := Cardinal.{u}) <| by
    /-
      c : Cardinal.{u}
      h : LT.lt c Cardinal.aleph0
      ⊢ Eq ↑(Cardinal.toNat c) ↑(Classical.choose ⋯)
    -/
    rw [cast_toNat_of_lt_aleph0 h, ← Classical.choose_spec (lt_aleph0.1 h)]
    /-
      🎉 no goals
    -/


                                                                                 /-
                                                                                   c : Cardinal.{u_1}
                                                                                   h : LE.le Cardinal.aleph0 c
                                                                                   ⊢ Eq (Cardinal.toNat c) 0
                                                                                 -/
theorem toNat_apply_of_aleph0_le {c : Cardinal} (h : ℵ₀ ≤ c) : toNat c = 0 := by simp [h]
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem cast_toNat_of_aleph0_le {c : Cardinal} (h : ℵ₀ ≤ c) : ↑(toNat c) = (0 : Cardinal) := by
  /-
    c : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 c
    ⊢ Eq (↑(Cardinal.toNat c)) 0
  -/
  rw [toNat_apply_of_aleph0_le h, Nat.cast_zero]
  /-
    🎉 no goals
  -/


theorem toNat_strictMonoOn : StrictMonoOn toNat (Iio ℵ₀) := by
  /-
    ⊢ StrictMonoOn (⇑Cardinal.toNat) (Set.Iio Cardinal.aleph0)
  -/
  simp only [← range_natCast, StrictMonoOn, forall_mem_range, toNat_natCast, Nat.cast_lt]
  /-
    ⊢ ∀ (i i_1 : Nat), LT.lt i i_1 → LT.lt i i_1
  -/
  exact fun _ _ ↦ id
  /-
    🎉 no goals
  -/


theorem toNat_monotoneOn : MonotoneOn toNat (Iio ℵ₀) := toNat_strictMonoOn.monotoneOn


theorem toNat_injOn : InjOn toNat (Iio ℵ₀) := toNat_strictMonoOn.injOn


/-- Two finite cardinals are equal
iff they are equal their `Cardinal.toNat` projections are equal. -/
theorem toNat_inj_of_lt_aleph0 (hc : c < ℵ₀) (hd : d < ℵ₀) :
    toNat c = toNat d ↔ c = d :=
  toNat_injOn.eq_iff hc hd


@[deprecated (since := "2024-12-29")] alias toNat_eq_iff_eq_of_lt_aleph0 := toNat_inj_of_lt_aleph0


theorem toNat_le_iff_le_of_lt_aleph0 (hc : c < ℵ₀) (hd : d < ℵ₀) :
    toNat c ≤ toNat d ↔ c ≤ d :=
  toNat_strictMonoOn.le_iff_le hc hd


theorem toNat_lt_iff_lt_of_lt_aleph0 (hc : c < ℵ₀) (hd : d < ℵ₀) :
    toNat c < toNat d ↔ c < d :=
  toNat_strictMonoOn.lt_iff_lt hc hd


@[gcongr]
theorem toNat_le_toNat (hcd : c ≤ d) (hd : d < ℵ₀) : toNat c ≤ toNat d :=
  toNat_monotoneOn (hcd.trans_lt hd) hd hcd


@[deprecated toNat_le_toNat (since := "2024-02-15")]
theorem toNat_le_of_le_of_lt_aleph0 (hd : d < ℵ₀) (hcd : c ≤ d) :
    toNat c ≤ toNat d :=
  toNat_le_toNat hcd hd


theorem toNat_lt_toNat (hcd : c < d) (hd : d < ℵ₀) : toNat c < toNat d :=
  toNat_strictMonoOn (hcd.trans hd) hd hcd


@[deprecated toNat_lt_toNat (since := "2024-02-15")]
theorem toNat_lt_of_lt_of_lt_aleph0 (hd : d < ℵ₀) (hcd : c < d) : toNat c < toNat d :=
  toNat_lt_toNat hcd hd


@[deprecated (since := "2024-02-15")] alias toNat_cast := toNat_natCast

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem toNat_ofNat (n : ℕ) [n.AtLeastTwo] :
    Cardinal.toNat (no_index (OfNat.ofNat n)) = OfNat.ofNat n :=
  toNat_natCast n


/-- `toNat` has a right-inverse: coercion. -/
theorem toNat_rightInverse : Function.RightInverse ((↑) : ℕ → Cardinal) toNat :=
  toNat_natCast


theorem toNat_surjective : Surjective toNat :=
  toNat_rightInverse.surjective


@[simp]
                                                                   /-
                                                                     α : Type u
                                                                     h : Infinite α
                                                                     ⊢ Eq (Cardinal.toNat (Cardinal.mk α)) 0
                                                                   -/
theorem mk_toNat_of_infinite [h : Infinite α] : toNat #α = 0 := by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem aleph0_toNat : toNat ℵ₀ = 0 :=
  toNat_apply_of_aleph0_le le_rfl


                                                                       /-
                                                                         α : Type u
                                                                         inst✝ : Fintype α
                                                                         ⊢ Eq (Cardinal.toNat (Cardinal.mk α)) (Fintype.card α)
                                                                       -/
theorem mk_toNat_eq_card [Fintype α] : toNat #α = Fintype.card α := by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem zero_toNat : toNat 0 = 0 := map_zero _


theorem one_toNat : toNat 1 = 1 := map_one _


theorem toNat_eq_iff {n : ℕ} (hn : n ≠ 0) : toNat c = n ↔ c = n := by
  /-
    c : Cardinal.{u}
    n : Nat
    hn : Ne n 0
    ⊢ Iff (Eq (Cardinal.toNat c) n) (Eq c ↑n)
  -/
  rw [← toNat_toENat, ENat.toNat_eq_iff hn, toENat_eq_nat]
  /-
    🎉 no goals
  -/


/-- A version of `toNat_eq_iff` for literals -/
theorem toNat_eq_ofNat {n : ℕ} [Nat.AtLeastTwo n] :
    toNat c = OfNat.ofNat n ↔ c = OfNat.ofNat n :=
  toNat_eq_iff <| OfNat.ofNat_ne_zero n


@[simp]
theorem toNat_eq_one : toNat c = 1 ↔ c = 1 := by
  /-
    c : Cardinal.{u}
    ⊢ Iff (Eq (Cardinal.toNat c) 1) (Eq c 1)
  -/
  rw [toNat_eq_iff one_ne_zero, Nat.cast_one]
  /-
    🎉 no goals
  -/


theorem toNat_eq_one_iff_unique : toNat #α = 1 ↔ Subsingleton α ∧ Nonempty α :=
  toNat_eq_one.trans eq_one_iff_unique


@[simp]
theorem toNat_lift (c : Cardinal.{v}) : toNat (lift.{u, v} c) = toNat c := by
  /-
    c : Cardinal.{v}
    ⊢ Eq (Cardinal.toNat (Cardinal.lift.{u, v} c)) (Cardinal.toNat c)
  -/
  simp only [← toNat_toENat, toENat_lift]
  /-
    🎉 no goals
  -/


theorem toNat_congr {β : Type v} (e : α ≃ β) : toNat #α = toNat #β := by
  -- Porting note: Inserted universe hint below
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    ⊢ Eq (Cardinal.toNat (Cardinal.mk α)) (Cardinal.toNat (Cardinal.mk β))
  -/
  rw [← toNat_lift, (lift_mk_eq.{_,_,v}).mpr ⟨e⟩, toNat_lift]
  /-
    🎉 no goals
  -/


theorem toNat_mul (x y : Cardinal) : toNat (x * y) = toNat x * toNat y := map_mul toNat x y


@[deprecated map_prod (since := "2024-02-15")]
theorem toNat_finset_prod (s : Finset α) (f : α → Cardinal) :
    toNat (∏ i ∈ s, f i) = ∏ i ∈ s, toNat (f i) :=
  map_prod toNat _ _


@[simp]
theorem toNat_add (hc : c < ℵ₀) (hd : d < ℵ₀) : toNat (c + d) = toNat c + toNat d := by
  /-
    c d : Cardinal.{u}
    hc : LT.lt c Cardinal.aleph0
    hd : LT.lt d Cardinal.aleph0
    ⊢ Eq (Cardinal.toNat (HAdd.hAdd c d)) (HAdd.hAdd (Cardinal.toNat c) (Cardinal. …
  -/
  lift c to ℕ using hc
  /-
    case intro
    d : Cardinal.{u}
    hd : LT.lt d Cardinal.aleph0
    c : Nat
    ⊢ Eq (Cardinal.toNat (HAdd.hAdd (↑c) d)) (HAdd.hAdd (Cardinal.toNat ↑c) (Cardi …
  -/
  lift d to ℕ using hd
  /-
    case intro.intro
    c d : Nat
    ⊢ Eq (Cardinal.toNat (HAdd.hAdd ↑c ↑d)) (HAdd.hAdd (Cardinal.toNat ↑c) (Cardin …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


@[simp]
theorem toNat_lift_add_lift {a : Cardinal.{u}} {b : Cardinal.{v}} (ha : a < ℵ₀) (hb : b < ℵ₀) :
    toNat (lift.{v} a + lift.{u} b) = toNat a + toNat b := by
  /-
    a : Cardinal.{u}
    b : Cardinal.{v}
    ha : LT.lt a Cardinal.aleph0
    hb : LT.lt b Cardinal.aleph0
    ⊢ Eq (Cardinal.toNat (HAdd.hAdd (Cardinal.lift.{v, u} a) (Cardinal.lift.{u, v} …
  -/
  simp [*]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-15")]
alias toNat_add_of_lt_aleph0 := toNat_lift_add_lift


