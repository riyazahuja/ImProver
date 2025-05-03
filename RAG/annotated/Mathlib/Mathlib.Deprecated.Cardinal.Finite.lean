/-- `PartENat.card α` is the cardinality of `α` as an extended natural number.
  If `α` is infinite, `PartENat.card α = ⊤`. -/
@[deprecated ENat.card (since := "2024-12-01")]
def card (α : Type*) : PartENat :=
  toPartENat (mk α)

-- This rest of this file is about the deprecated `PartENat.card`.

@[simp]
theorem card_eq_coe_fintype_card [Fintype α] : card α = Fintype.card α :=
  mk_toPartENat_eq_coe_card


@[simp]
theorem card_eq_top_of_infinite [Infinite α] : card α = ⊤ :=
  mk_toPartENat_of_infinite


@[simp]
theorem card_sum (α β : Type*) :
    PartENat.card (α ⊕ β) = PartENat.card α + PartENat.card β := by
  /-
    α : Type u_1
    β : Type u_2
    ⊢ Eq (PartENat.card (Sum α β)) (HAdd.hAdd (PartENat.card α) (PartENat.card β))
  -/
  simp only [PartENat.card, Cardinal.mk_sum, map_add, Cardinal.toPartENat_lift]
  /-
    🎉 no goals
  -/


theorem card_congr {α : Type*} {β : Type*} (f : α ≃ β) : PartENat.card α = PartENat.card β :=
  Cardinal.toPartENat_congr f


@[simp] lemma card_ulift (α : Type*) : card (ULift α) = card α := card_congr Equiv.ulift


@[simp] lemma card_plift (α : Type*) : card (PLift α) = card α := card_congr Equiv.plift


theorem card_image_of_injOn {α : Type u} {β : Type v} {f : α → β} {s : Set α} (h : Set.InjOn f s) :
    card (f '' s) = card s :=
  card_congr (Equiv.Set.imageOfInjOn f s h).symm


theorem card_image_of_injective {α : Type u} {β : Type v} (f : α → β) (s : Set α)
    (h : Function.Injective f) : card (f '' s) = card s := card_image_of_injOn h.injOn

-- Should I keep the 6 following lemmas ?
-- TODO: Add ofNat, zero, and one versions for simp confluence

@[simp]
theorem _root_.Cardinal.natCast_le_toPartENat_iff {n : ℕ} {c : Cardinal} :
    ↑n ≤ toPartENat c ↔ ↑n ≤ c := by
  /-
    n : Nat
    c : Cardinal.{u_1}
    ⊢ Iff (LE.le (↑n) (Cardinal.toPartENat c)) (LE.le (↑n) c)
  -/
  rw [← toPartENat_natCast n, toPartENat_le_iff_of_le_aleph0 (le_of_lt (nat_lt_aleph0 n))]
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.Cardinal.toPartENat_le_natCast_iff {c : Cardinal} {n : ℕ} :
    toPartENat c ≤ n ↔ c ≤ n := by
  /-
    c : Cardinal.{u_1}
    n : Nat
    ⊢ Iff (LE.le (Cardinal.toPartENat c) ↑n) (LE.le c ↑n)
  -/
  rw [← toPartENat_natCast n, toPartENat_le_iff_of_lt_aleph0 (nat_lt_aleph0 n)]
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.Cardinal.natCast_eq_toPartENat_iff {n : ℕ} {c : Cardinal} :
    ↑n = toPartENat c ↔ ↑n = c := by
  rw [le_antisymm_iff, le_antisymm_iff, Cardinal.toPartENat_le_natCast_iff,
    Cardinal.natCast_le_toPartENat_iff]


@[simp]
theorem _root_.Cardinal.toPartENat_eq_natCast_iff {c : Cardinal} {n : ℕ} :
    Cardinal.toPartENat c = n ↔ c = n := by
/-
  c : Cardinal.{u_1}
  n : Nat
  ⊢ Iff (Eq (Cardinal.toPartENat c) ↑n) (Eq c ↑n)
-/
rw [eq_comm, Cardinal.natCast_eq_toPartENat_iff, eq_comm]
/-
  🎉 no goals
-/


@[simp]
theorem _root_.Cardinal.natCast_lt_toPartENat_iff {n : ℕ} {c : Cardinal} :
    ↑n < toPartENat c ↔ ↑n < c := by
  /-
    n : Nat
    c : Cardinal.{u_1}
    ⊢ Iff (LT.lt (↑n) (Cardinal.toPartENat c)) (LT.lt (↑n) c)
  -/
  simp only [← not_le, Cardinal.toPartENat_le_natCast_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.Cardinal.toPartENat_lt_natCast_iff {n : ℕ} {c : Cardinal} :
    toPartENat c < ↑n ↔ c < ↑n := by
  /-
    n : Nat
    c : Cardinal.{u_1}
    ⊢ Iff (LT.lt (Cardinal.toPartENat c) ↑n) (LT.lt c ↑n)
  -/
  simp only [← not_le, Cardinal.natCast_le_toPartENat_iff]
  /-
    🎉 no goals
  -/


theorem card_eq_zero_iff_empty (α : Type*) : card α = 0 ↔ IsEmpty α := by
  /-
    α : Type u_1
    ⊢ Iff (Eq (PartENat.card α) 0) (IsEmpty α)
  -/
  rw [← Cardinal.mk_eq_zero_iff]
  /-
    α : Type u_1
    ⊢ Iff (Eq (PartENat.card α) 0) (Eq (Cardinal.mk α) 0)
  -/
  conv_rhs => rw [← Nat.cast_zero]
  /-
    α : Type u_1
    ⊢ Iff (Eq (PartENat.card α) 0) (Eq (Cardinal.mk α) ↑0)
  -/
  simp only [← Cardinal.toPartENat_eq_natCast_iff]
  /-
    α : Type u_1
    ⊢ Iff (Eq (PartENat.card α) 0) (Eq (Cardinal.toPartENat (Cardinal.mk α)) ↑0)
  -/
  simp only [PartENat.card, Nat.cast_zero]
  /-
    🎉 no goals
  -/


theorem card_le_one_iff_subsingleton (α : Type*) : card α ≤ 1 ↔ Subsingleton α := by
  /-
    α : Type u_1
    ⊢ Iff (LE.le (PartENat.card α) 1) (Subsingleton α)
  -/
  rw [← le_one_iff_subsingleton]
  /-
    α : Type u_1
    ⊢ Iff (LE.le (PartENat.card α) 1) (LE.le (Cardinal.mk α) 1)
  -/
  conv_rhs => rw [← Nat.cast_one]
  /-
    α : Type u_1
    ⊢ Iff (LE.le (PartENat.card α) 1) (LE.le (Cardinal.mk α) ↑1)
  -/
  rw [← Cardinal.toPartENat_le_natCast_iff]
  /-
    α : Type u_1
    ⊢ Iff (LE.le (PartENat.card α) 1) (LE.le (Cardinal.toPartENat (Cardinal.mk α)) …
  -/
  simp only [PartENat.card, Nat.cast_one]
  /-
    🎉 no goals
  -/


theorem one_lt_card_iff_nontrivial (α : Type*) : 1 < card α ↔ Nontrivial α := by
  /-
    α : Type u_1
    ⊢ Iff (LT.lt 1 (PartENat.card α)) (Nontrivial α)
  -/
  rw [← Cardinal.one_lt_iff_nontrivial]
  /-
    α : Type u_1
    ⊢ Iff (LT.lt 1 (PartENat.card α)) (LT.lt 1 (Cardinal.mk α))
  -/
  conv_rhs => rw [← Nat.cast_one]
  /-
    α : Type u_1
    ⊢ Iff (LT.lt 1 (PartENat.card α)) (LT.lt (↑1) (Cardinal.mk α))
  -/
  rw [← natCast_lt_toPartENat_iff]
  /-
    α : Type u_1
    ⊢ Iff (LT.lt 1 (PartENat.card α)) (LT.lt (↑1) (Cardinal.toPartENat (Cardinal.m …
  -/
  simp only [PartENat.card, Nat.cast_one]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated ENat.card_eq_coe_natCard (since := "2024-11-30")]
theorem card_eq_coe_natCard (α : Type*) [Finite α] : card α = Nat.card α := by
  /-
    α : Type u_1
    inst✝ : Finite α
    ⊢ Eq (PartENat.card α) ↑(Nat.card α)
  -/
  unfold PartENat.card
  /-
    α : Type u_1
    inst✝ : Finite α
    ⊢ Eq (Cardinal.toPartENat (Cardinal.mk α)) ↑(Nat.card α)
  -/
  apply symm
  /-
    case a
    α : Type u_1
    inst✝ : Finite α
    ⊢ Eq (↑(Nat.card α)) (Cardinal.toPartENat (Cardinal.mk α))
  -/
  rw [Cardinal.natCast_eq_toPartENat_iff]
  /-
    case a
    α : Type u_1
    inst✝ : Finite α
    ⊢ Eq (↑(Nat.card α)) (Cardinal.mk α)
  -/
  exact Finite.cast_card_eq_mk
  /-
    🎉 no goals
  -/



@[deprecated (since := "2024-05-25")] alias card_eq_coe_nat_card := card_eq_coe_natCard


