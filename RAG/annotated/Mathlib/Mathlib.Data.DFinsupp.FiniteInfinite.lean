instance DFinsupp.fintype {ι : Sort _} {π : ι → Sort _} [DecidableEq ι] [∀ i, Zero (π i)]
    [Fintype ι] [∀ i, Fintype (π i)] : Fintype (Π₀ i, π i) :=
  Fintype.ofEquiv (∀ i, π i) DFinsupp.equivFunOnFintype.symm


instance DFinsupp.infinite_of_left {ι : Sort _} {π : ι → Sort _} [∀ i, Nontrivial (π i)]
    [∀ i, Zero (π i)] [Infinite ι] : Infinite (Π₀ i, π i) := by
  /-
    ι✝ : Type u
    γ : Type w
    β : ι✝ → Type v
    β₁ : ι✝ → Type v₁
    β₂ : ι✝ → Type v₂
    ι : Type u_1
    π : ι → Type u_2
    inst✝² : ∀ (i : ι), Nontrivial (π i)
    inst✝¹ : (i : ι) → Zero (π i)
    inst✝ : Infinite ι
    ⊢ Infinite (DFinsupp fun i => π i)
  -/
  letI := Classical.decEq ι; choose m hm using fun i => exists_ne (0 : π i)
  /-
    ι✝ : Type u
    γ : Type w
    β : ι✝ → Type v
    β₁ : ι✝ → Type v₁
    β₂ : ι✝ → Type v₂
    ι : Type u_1
    π : ι → Type u_2
    inst✝² : ∀ (i : ι), Nontrivial (π i)
    inst✝¹ : (i : ι) → Zero (π i)
    inst✝ : Infinite ι
    this : DecidableEq ι := Classical.decEq ι
    m : (i : ι) → π i
    hm : ∀ (i : ι), Ne (m i) 0
    ⊢ Infinite (DFinsupp fun i => π i)
  -/
  exact Infinite.of_injective _ (DFinsupp.single_left_injective hm)
  /-
    🎉 no goals
  -/


/-- See `DFinsupp.infinite_of_right` for this in instance form, with the drawback that
it needs all `π i` to be infinite. -/
theorem DFinsupp.infinite_of_exists_right {ι : Sort _} {π : ι → Sort _} (i : ι) [Infinite (π i)]
    [∀ i, Zero (π i)] : Infinite (Π₀ i, π i) :=
  letI := Classical.decEq ι
  Infinite.of_injective (fun j => DFinsupp.single i j) DFinsupp.single_injective


/-- See `DFinsupp.infinite_of_exists_right` for the case that only one `π ι` is infinite. -/
instance DFinsupp.infinite_of_right {ι : Sort _} {π : ι → Sort _} [∀ i, Infinite (π i)]
    [∀ i, Zero (π i)] [Nonempty ι] : Infinite (Π₀ i, π i) :=
  DFinsupp.infinite_of_exists_right (Classical.arbitrary ι)


