theorem prod_dvd_iff {x y : G₁ × G₂} :
    x ∣ y ↔ x.1 ∣ y.1 ∧ x.2 ∣ y.2 := by
  /-
    G₁ : Type u_2
    G₂ : Type u_3
    inst✝¹ : Semigroup G₁
    inst✝ : Semigroup G₂
    x y : Prod G₁ G₂
    ⊢ Iff (Dvd.dvd x y) (And (Dvd.dvd x.1 y.1) (Dvd.dvd x.2 y.2))
  -/
  cases x; cases y
  simp only [dvd_def, Prod.exists, Prod.mk_mul_mk, Prod.mk.injEq,
    exists_and_left, exists_and_right, and_self, true_and]


@[simp]
theorem Prod.mk_dvd_mk {x₁ y₁ : G₁} {x₂ y₂ : G₂} :
    (x₁, x₂) ∣ (y₁, y₂) ↔ x₁ ∣ y₁ ∧ x₂ ∣ y₂ :=
  prod_dvd_iff


instance [DecompositionMonoid G₁] [DecompositionMonoid G₂] : DecompositionMonoid (G₁ × G₂) where
  primal a b c h := by
    /-
      ι : Type u_1
      G₁ : Type u_2
      G₂ : Type u_3
      G : ι → Type u_4
      inst✝⁴ : Semigroup G₁
      inst✝³ : Semigroup G₂
      inst✝² : (i : ι) → Semigroup (G i)
      inst✝¹ : DecompositionMonoid G₁
      inst✝ : DecompositionMonoid G₂
      a b c : Prod G₁ G₂
      h : Dvd.dvd a (HMul.hMul b c)
      ⊢ Exists fun a₁ => Exists fun a₂ => And (Dvd.dvd a₁ b) (And (Dvd.dvd a₂ c) (Eq …
    -/
    simp_rw [prod_dvd_iff] at h ⊢
    /-
      ι : Type u_1
      G₁ : Type u_2
      G₂ : Type u_3
      G : ι → Type u_4
      inst✝⁴ : Semigroup G₁
      inst✝³ : Semigroup G₂
      inst✝² : (i : ι) → Semigroup (G i)
      inst✝¹ : DecompositionMonoid G₁
      inst✝ : DecompositionMonoid G₂
      a b c : Prod G₁ G₂
      h : And (Dvd.dvd a.1 (HMul.hMul b c).1) (Dvd.dvd a.2 (HMul.hMul b c).2)
      ⊢ Exists fun a₁ => Exists fun a₂ => And (And (Dvd.dvd a₁.1 b.1) (Dvd.dvd a₁.2  …
    -/
    obtain ⟨a₁, a₁', h₁, h₁', eq₁⟩ := DecompositionMonoid.primal a.1 h.1
    /-
      case intro.intro.intro.intro
      ι : Type u_1
      G₁ : Type u_2
      G₂ : Type u_3
      G : ι → Type u_4
      inst✝⁴ : Semigroup G₁
      inst✝³ : Semigroup G₂
      inst✝² : (i : ι) → Semigroup (G i)
      inst✝¹ : DecompositionMonoid G₁
      inst✝ : DecompositionMonoid G₂
      a b c : Prod G₁ G₂
      h : And (Dvd.dvd a.1 (HMul.hMul b c).1) (Dvd.dvd a.2 (HMul.hMul b c).2)
      a₁ a₁' : G₁
      h₁ : Dvd.dvd a₁ b.1
      h₁' : Dvd.dvd a₁' c.1
      eq₁ : Eq a.1 (HMul.hMul a₁ a₁')
      ⊢ Exists fun a₁ => Exists fun a₂ => And (And (Dvd.dvd a₁.1 b.1) (Dvd.dvd a₁.2  …
    -/
    obtain ⟨a₂, a₂', h₂, h₂', eq₂⟩ := DecompositionMonoid.primal a.2 h.2
    -- aesop works here
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      ι : Type u_1
      G₁ : Type u_2
      G₂ : Type u_3
      G : ι → Type u_4
      inst✝⁴ : Semigroup G₁
      inst✝³ : Semigroup G₂
      inst✝² : (i : ι) → Semigroup (G i)
      inst✝¹ : DecompositionMonoid G₁
      inst✝ : DecompositionMonoid G₂
      a b c : Prod G₁ G₂
      h : And (Dvd.dvd a.1 (HMul.hMul b c).1) (Dvd.dvd a.2 (HMul.hMul b c).2)
      a₁ a₁' : G₁
      h₁ : Dvd.dvd a₁ b.1
      h₁' : Dvd.dvd a₁' c.1
      eq₁ : Eq a.1 (HMul.hMul a₁ a₁')
      a₂ a₂' : G₂
      h₂ : Dvd.dvd a₂ b.2
      h₂' : Dvd.dvd a₂' c.2
      eq₂ : Eq a.2 (HMul.hMul a₂ a₂')
      ⊢ Exists fun a₁ => Exists fun a₂ => And (And (Dvd.dvd a₁.1 b.1) (Dvd.dvd a₁.2  …
    -/
    exact ⟨(a₁, a₂), (a₁', a₂'), ⟨h₁, h₂⟩, ⟨h₁', h₂'⟩, Prod.ext eq₁ eq₂⟩
    /-
      🎉 no goals
    -/


theorem pi_dvd_iff {x y : ∀ i, G i} : x ∣ y ↔ ∀ i, x i ∣ y i := by
  /-
    ι : Type u_1
    G : ι → Type u_4
    inst✝ : (i : ι) → Semigroup (G i)
    x y : (i : ι) → G i
    ⊢ Iff (Dvd.dvd x y) (∀ (i : ι), Dvd.dvd (x i) (y i))
  -/
  simp_rw [dvd_def, funext_iff, Classical.skolem]; rfl
                                                   /-
                                                     🎉 no goals
                                                   -/


instance [∀ i, DecompositionMonoid (G i)] : DecompositionMonoid (∀ i, G i) where
  primal a b c h := by
    /-
      ι : Type u_1
      G₁ : Type u_2
      G₂ : Type u_3
      G : ι → Type u_4
      inst✝³ : Semigroup G₁
      inst✝² : Semigroup G₂
      inst✝¹ : (i : ι) → Semigroup (G i)
      inst✝ : ∀ (i : ι), DecompositionMonoid (G i)
      a b c : (i : ι) → G i
      h : Dvd.dvd a (HMul.hMul b c)
      ⊢ Exists fun a₁ => Exists fun a₂ => And (Dvd.dvd a₁ b) (And (Dvd.dvd a₂ c) (Eq …
    -/
    simp_rw [pi_dvd_iff] at h ⊢
    /-
      ι : Type u_1
      G₁ : Type u_2
      G₂ : Type u_3
      G : ι → Type u_4
      inst✝³ : Semigroup G₁
      inst✝² : Semigroup G₂
      inst✝¹ : (i : ι) → Semigroup (G i)
      inst✝ : ∀ (i : ι), DecompositionMonoid (G i)
      a b c : (i : ι) → G i
      h : ∀ (i : ι), Dvd.dvd (a i) (HMul.hMul b c i)
      ⊢ Exists fun a₁ => Exists fun a₂ => And (∀ (i : ι), Dvd.dvd (a₁ i) (b i)) (And …
    -/
    choose a₁ a₂ h₁ h₂ eq using fun i ↦ DecompositionMonoid.primal _ (h i)
    /-
      ι : Type u_1
      G₁ : Type u_2
      G₂ : Type u_3
      G : ι → Type u_4
      inst✝³ : Semigroup G₁
      inst✝² : Semigroup G₂
      inst✝¹ : (i : ι) → Semigroup (G i)
      inst✝ : ∀ (i : ι), DecompositionMonoid (G i)
      a b c : (i : ι) → G i
      h : ∀ (i : ι), Dvd.dvd (a i) (HMul.hMul b c i)
      a₁ a₂ : (i : ι) → G i
      h₁ : ∀ (i : ι), Dvd.dvd (a₁ i) (b i)
      h₂ : ∀ (i : ι), Dvd.dvd (a₂ i) (c i)
      eq : ∀ (i : ι), Eq (a i) (HMul.hMul (a₁ i) (a₂ i))
      ⊢ Exists fun a₁ => Exists fun a₂ => And (∀ (i : ι), Dvd.dvd (a₁ i) (b i)) (And …
    -/
    exact ⟨a₁, a₂, h₁, h₂, funext eq⟩
    /-
      🎉 no goals
    -/

