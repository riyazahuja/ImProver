include hg hρ h₂ hσ hι₃ in
lemma SnakeLemma.δ_aux (x : K₃) : g₁ (ρ (i₂ (σ (ι₃ x)))) = i₂ (σ (ι₃ x)) := by
  obtain ⟨d, hd⟩ : i₂ (σ (ι₃ x)) ∈ range g₁ := by
    rw [← hg.linearMap_ker_eq, mem_ker, show g₂ (i₂ _) = i₃ (f₂ _) from DFunLike.congr_fun h₂ _,
      ← @comp_apply _ _ _ f₂ σ, hσ, id_eq, ← i₃.comp_apply,
      hι₃.linearMap_comp_eq_zero, zero_apply]
  /-
    case intro
    R : Type u_3
    inst✝¹² : CommRing R
    M₂ : Type u_4
    M₃ : Type u_5
    N₁ : Type u_2
    N₂ : Type u_1
    N₃ : Type ?u.16465
    inst✝¹¹ : AddCommGroup M₂
    inst✝¹⁰ : Module R M₂
    inst✝⁹ : AddCommGroup M₃
    inst✝⁸ : Module R M₃
    inst✝⁷ : AddCommGroup N₁
    inst✝⁶ : Module R N₁
    inst✝⁵ : AddCommGroup N₂
    inst✝⁴ : Module R N₂
    inst✝³ : AddCommGroup N₃
    inst✝² : Module R N₃
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    i₃ : LinearMap (RingHom.id R) M₃ N₃
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hg : Function.Exact ⇑g₁ ⇑g₂
    h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
    σ : M₃ → M₂
    hσ : Eq (Function.comp (⇑f₂) σ) id
    ρ : N₂ → N₁
    hρ : Eq (Function.comp ρ ⇑g₁) id
    K₃ : Type u_6
    inst✝¹ : AddCommGroup K₃
    inst✝ : Module R K₃
    ι₃ : LinearMap (RingHom.id R) K₃ M₃
    hι₃ : Function.Exact ⇑ι₃ ⇑i₃
    x : K₃
    d : N₁
    hd : Eq (g₁ d) (i₂ (σ (ι₃ x)))
    ⊢ Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
  -/
  rw [← hd, ← ρ.comp_apply, hρ, id_eq]
  /-
    🎉 no goals
  -/


include hf h₁ hρ hπ₁ in
lemma SnakeLemma.eq_of_eq (x : K₃)
    (y₁) (hy₁ : f₂ y₁ = ι₃ x) (z₁) (hz₁ : g₁ z₁ = i₂ y₁)
    (y₂) (hy₂ : f₂ y₂ = ι₃ x) (z₂) (hz₂ : g₁ z₂ = i₂ y₂) : π₁ z₁ = π₁ z₂ := by
  /-
    R : Type u_3
    inst✝¹⁴ : CommRing R
    M₁ : Type ?u.36806
    M₂ : Type u_2
    M₃ : Type u_1
    N₁ : Type u_6
    N₂ : Type u_5
    inst✝¹³ : AddCommGroup M₁
    inst✝¹² : Module R M₁
    inst✝¹¹ : AddCommGroup M₂
    inst✝¹⁰ : Module R M₂
    inst✝⁹ : AddCommGroup M₃
    inst✝⁸ : Module R M₃
    inst✝⁷ : AddCommGroup N₁
    inst✝⁶ : Module R N₁
    inst✝⁵ : AddCommGroup N₂
    inst✝⁴ : Module R N₂
    i₁ : LinearMap (RingHom.id R) M₁ N₁
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    hf : Function.Exact ⇑f₁ ⇑f₂
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
    ρ : N₂ → N₁
    hρ : Eq (Function.comp ρ ⇑g₁) id
    K₃ : Type u_4
    C₁ : Type u_7
    inst✝³ : AddCommGroup K₃
    inst✝² : Module R K₃
    inst✝¹ : AddCommGroup C₁
    inst✝ : Module R C₁
    ι₃ : LinearMap (RingHom.id R) K₃ M₃
    π₁ : LinearMap (RingHom.id R) N₁ C₁
    hπ₁ : Function.Exact ⇑i₁ ⇑π₁
    x : K₃
    y₁ : M₂
    hy₁ : Eq (f₂ y₁) (ι₃ x)
    z₁ : N₁
    hz₁ : Eq (g₁ z₁) (i₂ y₁)
    y₂ : M₂
    hy₂ : Eq (f₂ y₂) (ι₃ x)
    z₂ : N₁
    hz₂ : Eq (g₁ z₂) (i₂ y₂)
    ⊢ Eq (π₁ z₁) (π₁ z₂)
  -/
  have := sub_eq_zero.mpr (hy₁.trans hy₂.symm)
  /-
    R : Type u_3
    inst✝¹⁴ : CommRing R
    M₁ : Type ?u.36806
    M₂ : Type u_2
    M₃ : Type u_1
    N₁ : Type u_6
    N₂ : Type u_5
    inst✝¹³ : AddCommGroup M₁
    inst✝¹² : Module R M₁
    inst✝¹¹ : AddCommGroup M₂
    inst✝¹⁰ : Module R M₂
    inst✝⁹ : AddCommGroup M₃
    inst✝⁸ : Module R M₃
    inst✝⁷ : AddCommGroup N₁
    inst✝⁶ : Module R N₁
    inst✝⁵ : AddCommGroup N₂
    inst✝⁴ : Module R N₂
    i₁ : LinearMap (RingHom.id R) M₁ N₁
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    hf : Function.Exact ⇑f₁ ⇑f₂
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
    ρ : N₂ → N₁
    hρ : Eq (Function.comp ρ ⇑g₁) id
    K₃ : Type u_4
    C₁ : Type u_7
    inst✝³ : AddCommGroup K₃
    inst✝² : Module R K₃
    inst✝¹ : AddCommGroup C₁
    inst✝ : Module R C₁
    ι₃ : LinearMap (RingHom.id R) K₃ M₃
    π₁ : LinearMap (RingHom.id R) N₁ C₁
    hπ₁ : Function.Exact ⇑i₁ ⇑π₁
    x : K₃
    y₁ : M₂
    hy₁ : Eq (f₂ y₁) (ι₃ x)
    z₁ : N₁
    hz₁ : Eq (g₁ z₁) (i₂ y₁)
    y₂ : M₂
    hy₂ : Eq (f₂ y₂) (ι₃ x)
    z₂ : N₁
    hz₂ : Eq (g₁ z₂) (i₂ y₂)
    this : Eq (HSub.hSub (f₂ y₁) (f₂ y₂)) 0
    ⊢ Eq (π₁ z₁) (π₁ z₂)
  -/
  rw [← map_sub, hf] at this
  /-
    R : Type u_3
    inst✝¹⁴ : CommRing R
    M₁ : Type ?u.36806
    M₂ : Type u_2
    M₃ : Type u_1
    N₁ : Type u_6
    N₂ : Type u_5
    inst✝¹³ : AddCommGroup M₁
    inst✝¹² : Module R M₁
    inst✝¹¹ : AddCommGroup M₂
    inst✝¹⁰ : Module R M₂
    inst✝⁹ : AddCommGroup M₃
    inst✝⁸ : Module R M₃
    inst✝⁷ : AddCommGroup N₁
    inst✝⁶ : Module R N₁
    inst✝⁵ : AddCommGroup N₂
    inst✝⁴ : Module R N₂
    i₁ : LinearMap (RingHom.id R) M₁ N₁
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    hf : Function.Exact ⇑f₁ ⇑f₂
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
    ρ : N₂ → N₁
    hρ : Eq (Function.comp ρ ⇑g₁) id
    K₃ : Type u_4
    C₁ : Type u_7
    inst✝³ : AddCommGroup K₃
    inst✝² : Module R K₃
    inst✝¹ : AddCommGroup C₁
    inst✝ : Module R C₁
    ι₃ : LinearMap (RingHom.id R) K₃ M₃
    π₁ : LinearMap (RingHom.id R) N₁ C₁
    hπ₁ : Function.Exact ⇑i₁ ⇑π₁
    x : K₃
    y₁ : M₂
    hy₁ : Eq (f₂ y₁) (ι₃ x)
    z₁ : N₁
    hz₁ : Eq (g₁ z₁) (i₂ y₁)
    y₂ : M₂
    hy₂ : Eq (f₂ y₂) (ι₃ x)
    z₂ : N₁
    hz₂ : Eq (g₁ z₂) (i₂ y₂)
    this : Membership.mem (Set.range ⇑f₁) (HSub.hSub y₁ y₂)
    ⊢ Eq (π₁ z₁) (π₁ z₂)
  -/
  obtain ⟨d, hd⟩ := this
  rw [← eq_sub_iff_add_eq.mp hd, map_add, ← hz₂, ← sub_eq_iff_eq_add, ← map_sub,
    ← i₂.comp_apply, ← h₁, LinearMap.comp_apply,
    (HasLeftInverse.injective ⟨ρ, congr_fun hρ⟩).eq_iff] at hz₁
  /-
    case intro
    R : Type u_3
    inst✝¹⁴ : CommRing R
    M₁ : Type ?u.36806
    M₂ : Type u_2
    M₃ : Type u_1
    N₁ : Type u_6
    N₂ : Type u_5
    inst✝¹³ : AddCommGroup M₁
    inst✝¹² : Module R M₁
    inst✝¹¹ : AddCommGroup M₂
    inst✝¹⁰ : Module R M₂
    inst✝⁹ : AddCommGroup M₃
    inst✝⁸ : Module R M₃
    inst✝⁷ : AddCommGroup N₁
    inst✝⁶ : Module R N₁
    inst✝⁵ : AddCommGroup N₂
    inst✝⁴ : Module R N₂
    i₁ : LinearMap (RingHom.id R) M₁ N₁
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    hf : Function.Exact ⇑f₁ ⇑f₂
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
    ρ : N₂ → N₁
    hρ : Eq (Function.comp ρ ⇑g₁) id
    K₃ : Type u_4
    C₁ : Type u_7
    inst✝³ : AddCommGroup K₃
    inst✝² : Module R K₃
    inst✝¹ : AddCommGroup C₁
    inst✝ : Module R C₁
    ι₃ : LinearMap (RingHom.id R) K₃ M₃
    π₁ : LinearMap (RingHom.id R) N₁ C₁
    hπ₁ : Function.Exact ⇑i₁ ⇑π₁
    x : K₃
    y₁ : M₂
    hy₁ : Eq (f₂ y₁) (ι₃ x)
    z₁ : N₁
    y₂ : M₂
    hy₂ : Eq (f₂ y₂) (ι₃ x)
    z₂ : N₁
    hz₂ : Eq (g₁ z₂) (i₂ y₂)
    d : M₁
    hz₁ : Eq (HSub.hSub z₁ z₂) (i₁ d)
    hd : Eq (f₁ d) (HSub.hSub y₁ y₂)
    ⊢ Eq (π₁ z₁) (π₁ z₂)
  -/
  rw [← sub_eq_zero, ← map_sub, hz₁, hπ₁]
  /-
    case intro
    R : Type u_3
    inst✝¹⁴ : CommRing R
    M₁ : Type ?u.36806
    M₂ : Type u_2
    M₃ : Type u_1
    N₁ : Type u_6
    N₂ : Type u_5
    inst✝¹³ : AddCommGroup M₁
    inst✝¹² : Module R M₁
    inst✝¹¹ : AddCommGroup M₂
    inst✝¹⁰ : Module R M₂
    inst✝⁹ : AddCommGroup M₃
    inst✝⁸ : Module R M₃
    inst✝⁷ : AddCommGroup N₁
    inst✝⁶ : Module R N₁
    inst✝⁵ : AddCommGroup N₂
    inst✝⁴ : Module R N₂
    i₁ : LinearMap (RingHom.id R) M₁ N₁
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    hf : Function.Exact ⇑f₁ ⇑f₂
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
    ρ : N₂ → N₁
    hρ : Eq (Function.comp ρ ⇑g₁) id
    K₃ : Type u_4
    C₁ : Type u_7
    inst✝³ : AddCommGroup K₃
    inst✝² : Module R K₃
    inst✝¹ : AddCommGroup C₁
    inst✝ : Module R C₁
    ι₃ : LinearMap (RingHom.id R) K₃ M₃
    π₁ : LinearMap (RingHom.id R) N₁ C₁
    hπ₁ : Function.Exact ⇑i₁ ⇑π₁
    x : K₃
    y₁ : M₂
    hy₁ : Eq (f₂ y₁) (ι₃ x)
    z₁ : N₁
    y₂ : M₂
    hy₂ : Eq (f₂ y₂) (ι₃ x)
    z₂ : N₁
    hz₂ : Eq (g₁ z₂) (i₂ y₂)
    d : M₁
    hz₁ : Eq (HSub.hSub z₁ z₂) (i₁ d)
    hd : Eq (f₁ d) (HSub.hSub y₁ y₂)
    ⊢ Membership.mem (Set.range ⇑i₁) (i₁ d)
  -/
  exact ⟨_, rfl⟩
  /-
    🎉 no goals
  -/


/--
**Snake Lemma**
Supppose we have an exact commutative diagram
```
                K₃
                |
                ι₃
                ↓
M₁ -f₁→ M₂ -f₂→ M₃
|       |       |
i₁      i₂      i₃
↓       ↓       ↓
N₁ -g₁→ N₂ -g₂→ N₃
|
π₁
↓
C₁

```
such that `f₂` is surjective with a (set-theoretic) section `σ`, `g₁` is injective with a
(set-theoretic) retraction `ρ`,
then the map `π₁ ∘ ρ ∘ i₂ ∘ σ ∘ ι₃` is a linear map from `K₃` to `C₁`.

Also see `SnakeLemma.δ'` for a noncomputable version
that does not require an explicit section and retraction.
-/
def SnakeLemma.δ : K₃ →ₗ[R] C₁ :=
  haveI H₁ : ∀ x, f₂ (σ x) = x := congr_fun hσ
  haveI H₂ := δ_aux i₂ i₃ f₂ g₁ g₂ hg h₂ σ hσ ρ hρ ι₃ hι₃
  { toFun := fun x ↦ π₁ (ρ (i₂ (σ (ι₃ x))))
    map_add' := fun x y ↦ by
      /-
        R : Type ?u.63289
        inst✝²⁰ : CommRing R
        M₁ : Type ?u.63310
        M₂ : Type ?u.63821
        M₃ : Type ?u.64298
        N₁ : Type ?u.64775
        N₂ : Type ?u.65252
        N₃ : Type ?u.65729
        inst✝¹⁹ : AddCommGroup M₁
        inst✝¹⁸ : Module R M₁
        inst✝¹⁷ : AddCommGroup M₂
        inst✝¹⁶ : Module R M₂
        inst✝¹⁵ : AddCommGroup M₃
        inst✝¹⁴ : Module R M₃
        inst✝¹³ : AddCommGroup N₁
        inst✝¹² : Module R N₁
        inst✝¹¹ : AddCommGroup N₂
        inst✝¹⁰ : Module R N₂
        inst✝⁹ : AddCommGroup N₃
        inst✝⁸ : Module R N₃
        i₁ : LinearMap (RingHom.id R) M₁ N₁
        i₂ : LinearMap (RingHom.id R) M₂ N₂
        i₃ : LinearMap (RingHom.id R) M₃ N₃
        f₁ : LinearMap (RingHom.id R) M₁ M₂
        f₂ : LinearMap (RingHom.id R) M₂ M₃
        hf : Function.Exact ⇑f₁ ⇑f₂
        g₁ : LinearMap (RingHom.id R) N₁ N₂
        g₂ : LinearMap (RingHom.id R) N₂ N₃
        hg : Function.Exact ⇑g₁ ⇑g₂
        h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
        h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
        σ : M₃ → M₂
        hσ : Eq (Function.comp (⇑f₂) σ) id
        ρ : N₂ → N₁
        hρ : Eq (Function.comp ρ ⇑g₁) id
        K₂ : Type ?u.70524
        K₃ : Type ?u.71001
        C₁ : Type ?u.71478
        C₂ : Type ?u.71955
        inst✝⁷ : AddCommGroup K₂
        inst✝⁶ : Module R K₂
        inst✝⁵ : AddCommGroup K₃
        inst✝⁴ : Module R K₃
        inst✝³ : AddCommGroup C₁
        inst✝² : Module R C₁
        inst✝¹ : AddCommGroup C₂
        inst✝ : Module R C₂
        ι₂ : LinearMap (RingHom.id R) K₂ M₂
        hι₂ : Function.Exact ⇑ι₂ ⇑i₂
        ι₃ : LinearMap (RingHom.id R) K₃ M₃
        hι₃ : Function.Exact ⇑ι₃ ⇑i₃
        π₁ : LinearMap (RingHom.id R) N₁ C₁
        hπ₁ : Function.Exact ⇑i₁ ⇑π₁
        π₂ : LinearMap (RingHom.id R) N₂ C₂
        hπ₂ : Function.Exact ⇑i₂ ⇑π₂
        H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
        H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
        x y : K₃
        ⊢ Eq ((fun x => π₁ (ρ (i₂ (σ (ι₃ x))))) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x => …
      -/
      rw [← map_add]
      exact eq_of_eq i₁ i₂ f₁ f₂ hf g₁ h₁ ρ hρ ι₃ π₁ hπ₁ (x + y) _ (H₁ _) _ (H₂ _)
        (σ (ι₃ x) + σ (ι₃ y)) (by simp only [map_add, H₁]) _ (by simp only [map_add, H₂])
    map_smul' := fun r x ↦ by
      /-
        R : Type ?u.63289
        inst✝²⁰ : CommRing R
        M₁ : Type ?u.63310
        M₂ : Type ?u.63821
        M₃ : Type ?u.64298
        N₁ : Type ?u.64775
        N₂ : Type ?u.65252
        N₃ : Type ?u.65729
        inst✝¹⁹ : AddCommGroup M₁
        inst✝¹⁸ : Module R M₁
        inst✝¹⁷ : AddCommGroup M₂
        inst✝¹⁶ : Module R M₂
        inst✝¹⁵ : AddCommGroup M₃
        inst✝¹⁴ : Module R M₃
        inst✝¹³ : AddCommGroup N₁
        inst✝¹² : Module R N₁
        inst✝¹¹ : AddCommGroup N₂
        inst✝¹⁰ : Module R N₂
        inst✝⁹ : AddCommGroup N₃
        inst✝⁸ : Module R N₃
        i₁ : LinearMap (RingHom.id R) M₁ N₁
        i₂ : LinearMap (RingHom.id R) M₂ N₂
        i₃ : LinearMap (RingHom.id R) M₃ N₃
        f₁ : LinearMap (RingHom.id R) M₁ M₂
        f₂ : LinearMap (RingHom.id R) M₂ M₃
        hf : Function.Exact ⇑f₁ ⇑f₂
        g₁ : LinearMap (RingHom.id R) N₁ N₂
        g₂ : LinearMap (RingHom.id R) N₂ N₃
        hg : Function.Exact ⇑g₁ ⇑g₂
        h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
        h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
        σ : M₃ → M₂
        hσ : Eq (Function.comp (⇑f₂) σ) id
        ρ : N₂ → N₁
        hρ : Eq (Function.comp ρ ⇑g₁) id
        K₂ : Type ?u.70524
        K₃ : Type ?u.71001
        C₁ : Type ?u.71478
        C₂ : Type ?u.71955
        inst✝⁷ : AddCommGroup K₂
        inst✝⁶ : Module R K₂
        inst✝⁵ : AddCommGroup K₃
        inst✝⁴ : Module R K₃
        inst✝³ : AddCommGroup C₁
        inst✝² : Module R C₁
        inst✝¹ : AddCommGroup C₂
        inst✝ : Module R C₂
        ι₂ : LinearMap (RingHom.id R) K₂ M₂
        hι₂ : Function.Exact ⇑ι₂ ⇑i₂
        ι₃ : LinearMap (RingHom.id R) K₃ M₃
        hι₃ : Function.Exact ⇑ι₃ ⇑i₃
        π₁ : LinearMap (RingHom.id R) N₁ C₁
        hπ₁ : Function.Exact ⇑i₁ ⇑π₁
        π₂ : LinearMap (RingHom.id R) N₂ C₂
        hπ₂ : Function.Exact ⇑i₂ ⇑π₂
        H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
        H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
        r : R
        x : K₃
        ⊢ Eq ({ toFun := fun x => π₁ (ρ (i₂ (σ (ι₃ x)))), map_add' := ⋯ }.toFun (HSMul …
      -/
      simp only [← map_smul, RingHom.id_apply]
      apply eq_of_eq i₁ i₂ f₁ f₂ hf g₁ h₁ ρ hρ ι₃ π₁ hπ₁ (r • x) _ (H₁ _) _ (H₂ _)
        (r • σ (ι₃ x)) (by simp only [map_smul, H₁]) _ (by simp only [map_smul, H₂]) }


lemma SnakeLemma.δ_eq (x : K₃) (y) (hy : f₂ y = ι₃ x) (z) (hz : g₁ z = i₂ y) :
    δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ ι₃ hι₃ π₁ hπ₁ x = π₁ z :=
  eq_of_eq i₁ i₂ f₁ f₂ hf g₁ h₁ ρ hρ ι₃ π₁ hπ₁ x _ (congr_fun hσ _) _
    (δ_aux i₂ i₃ f₂ g₁ g₂ hg h₂ σ hσ ρ hρ ι₃ hι₃ _) y hy z hz


include hι₂ in
/--
Supppose we have an exact commutative diagram
```
        K₂ -F-→ K₃
        |       |
        ι₂      ι₃
        ↓       ↓
M₁ -f₁→ M₂ -f₂→ M₃
|       |       |
i₁      i₂      i₃
↓       ↓       ↓
N₁ -g₁→ N₂ -g₂→ N₃
|
π₁
↓
C₁

```
such that `f₂` is surjective with a (set-theoretic) section `σ`, `g₁` is injective with a
(set-theoretic) retraction `ρ`, and `ι₃` is injective, then `K₂ -F→ K₂ -δ→ C₁` is exact.
-/
lemma SnakeLemma.exact_δ_right (F : K₂ →ₗ[R] K₃) (hF : f₂.comp ι₂ = ι₃.comp F)
    (h : Injective ι₃) :
    Exact F (δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ ι₃ hι₃ π₁ hπ₁) := by
  /-
    R : Type u_1
    inst✝¹⁸ : CommRing R
    M₁ : Type u_7
    M₂ : Type u_5
    M₃ : Type u_4
    N₁ : Type u_8
    N₂ : Type u_9
    N₃ : Type u_10
    inst✝¹⁷ : AddCommGroup M₁
    inst✝¹⁶ : Module R M₁
    inst✝¹⁵ : AddCommGroup M₂
    inst✝¹⁴ : Module R M₂
    inst✝¹³ : AddCommGroup M₃
    inst✝¹² : Module R M₃
    inst✝¹¹ : AddCommGroup N₁
    inst✝¹⁰ : Module R N₁
    inst✝⁹ : AddCommGroup N₂
    inst✝⁸ : Module R N₂
    inst✝⁷ : AddCommGroup N₃
    inst✝⁶ : Module R N₃
    i₁ : LinearMap (RingHom.id R) M₁ N₁
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    i₃ : LinearMap (RingHom.id R) M₃ N₃
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    hf : Function.Exact ⇑f₁ ⇑f₂
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hg : Function.Exact ⇑g₁ ⇑g₂
    h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
    h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
    σ : M₃ → M₂
    hσ : Eq (Function.comp (⇑f₂) σ) id
    ρ : N₂ → N₁
    hρ : Eq (Function.comp ρ ⇑g₁) id
    K₂ : Type u_2
    K₃ : Type u_3
    C₁ : Type u_6
    inst✝⁵ : AddCommGroup K₂
    inst✝⁴ : Module R K₂
    inst✝³ : AddCommGroup K₃
    inst✝² : Module R K₃
    inst✝¹ : AddCommGroup C₁
    inst✝ : Module R C₁
    ι₂ : LinearMap (RingHom.id R) K₂ M₂
    hι₂ : Function.Exact ⇑ι₂ ⇑i₂
    ι₃ : LinearMap (RingHom.id R) K₃ M₃
    hι₃ : Function.Exact ⇑ι₃ ⇑i₃
    π₁ : LinearMap (RingHom.id R) N₁ C₁
    hπ₁ : Function.Exact ⇑i₁ ⇑π₁
    F : LinearMap (RingHom.id R) K₂ K₃
    hF : Eq (f₂.comp ι₂) (ι₃.comp F)
    h : Function.Injective ⇑ι₃
    ⊢ Function.Exact ⇑F ⇑(SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ  …
  -/
  haveI H₁ : ∀ x, f₂ (σ x) = x := congr_fun hσ
  /-
    R : Type u_1
    inst✝¹⁸ : CommRing R
    M₁ : Type u_7
    M₂ : Type u_5
    M₃ : Type u_4
    N₁ : Type u_8
    N₂ : Type u_9
    N₃ : Type u_10
    inst✝¹⁷ : AddCommGroup M₁
    inst✝¹⁶ : Module R M₁
    inst✝¹⁵ : AddCommGroup M₂
    inst✝¹⁴ : Module R M₂
    inst✝¹³ : AddCommGroup M₃
    inst✝¹² : Module R M₃
    inst✝¹¹ : AddCommGroup N₁
    inst✝¹⁰ : Module R N₁
    inst✝⁹ : AddCommGroup N₂
    inst✝⁸ : Module R N₂
    inst✝⁷ : AddCommGroup N₃
    inst✝⁶ : Module R N₃
    i₁ : LinearMap (RingHom.id R) M₁ N₁
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    i₃ : LinearMap (RingHom.id R) M₃ N₃
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    hf : Function.Exact ⇑f₁ ⇑f₂
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hg : Function.Exact ⇑g₁ ⇑g₂
    h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
    h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
    σ : M₃ → M₂
    hσ : Eq (Function.comp (⇑f₂) σ) id
    ρ : N₂ → N₁
    hρ : Eq (Function.comp ρ ⇑g₁) id
    K₂ : Type u_2
    K₃ : Type u_3
    C₁ : Type u_6
    inst✝⁵ : AddCommGroup K₂
    inst✝⁴ : Module R K₂
    inst✝³ : AddCommGroup K₃
    inst✝² : Module R K₃
    inst✝¹ : AddCommGroup C₁
    inst✝ : Module R C₁
    ι₂ : LinearMap (RingHom.id R) K₂ M₂
    hι₂ : Function.Exact ⇑ι₂ ⇑i₂
    ι₃ : LinearMap (RingHom.id R) K₃ M₃
    hι₃ : Function.Exact ⇑ι₃ ⇑i₃
    π₁ : LinearMap (RingHom.id R) N₁ C₁
    hπ₁ : Function.Exact ⇑i₁ ⇑π₁
    F : LinearMap (RingHom.id R) K₂ K₃
    hF : Eq (f₂.comp ι₂) (ι₃.comp F)
    h : Function.Injective ⇑ι₃
    H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
    ⊢ Function.Exact ⇑F ⇑(SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ  …
  -/
  haveI H₂ := δ_aux i₂ i₃ f₂ g₁ g₂ hg h₂ σ hσ ρ hρ ι₃ hι₃
  /-
    R : Type u_1
    inst✝¹⁸ : CommRing R
    M₁ : Type u_7
    M₂ : Type u_5
    M₃ : Type u_4
    N₁ : Type u_8
    N₂ : Type u_9
    N₃ : Type u_10
    inst✝¹⁷ : AddCommGroup M₁
    inst✝¹⁶ : Module R M₁
    inst✝¹⁵ : AddCommGroup M₂
    inst✝¹⁴ : Module R M₂
    inst✝¹³ : AddCommGroup M₃
    inst✝¹² : Module R M₃
    inst✝¹¹ : AddCommGroup N₁
    inst✝¹⁰ : Module R N₁
    inst✝⁹ : AddCommGroup N₂
    inst✝⁸ : Module R N₂
    inst✝⁷ : AddCommGroup N₃
    inst✝⁶ : Module R N₃
    i₁ : LinearMap (RingHom.id R) M₁ N₁
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    i₃ : LinearMap (RingHom.id R) M₃ N₃
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    hf : Function.Exact ⇑f₁ ⇑f₂
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hg : Function.Exact ⇑g₁ ⇑g₂
    h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
    h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
    σ : M₃ → M₂
    hσ : Eq (Function.comp (⇑f₂) σ) id
    ρ : N₂ → N₁
    hρ : Eq (Function.comp ρ ⇑g₁) id
    K₂ : Type u_2
    K₃ : Type u_3
    C₁ : Type u_6
    inst✝⁵ : AddCommGroup K₂
    inst✝⁴ : Module R K₂
    inst✝³ : AddCommGroup K₃
    inst✝² : Module R K₃
    inst✝¹ : AddCommGroup C₁
    inst✝ : Module R C₁
    ι₂ : LinearMap (RingHom.id R) K₂ M₂
    hι₂ : Function.Exact ⇑ι₂ ⇑i₂
    ι₃ : LinearMap (RingHom.id R) K₃ M₃
    hι₃ : Function.Exact ⇑ι₃ ⇑i₃
    π₁ : LinearMap (RingHom.id R) N₁ C₁
    hπ₁ : Function.Exact ⇑i₁ ⇑π₁
    F : LinearMap (RingHom.id R) K₂ K₃
    hF : Eq (f₂.comp ι₂) (ι₃.comp F)
    h : Function.Injective ⇑ι₃
    H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
    H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
    ⊢ Function.Exact ⇑F ⇑(SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ  …
  -/
  intro x
  /-
    R : Type u_1
    inst✝¹⁸ : CommRing R
    M₁ : Type u_7
    M₂ : Type u_5
    M₃ : Type u_4
    N₁ : Type u_8
    N₂ : Type u_9
    N₃ : Type u_10
    inst✝¹⁷ : AddCommGroup M₁
    inst✝¹⁶ : Module R M₁
    inst✝¹⁵ : AddCommGroup M₂
    inst✝¹⁴ : Module R M₂
    inst✝¹³ : AddCommGroup M₃
    inst✝¹² : Module R M₃
    inst✝¹¹ : AddCommGroup N₁
    inst✝¹⁰ : Module R N₁
    inst✝⁹ : AddCommGroup N₂
    inst✝⁸ : Module R N₂
    inst✝⁷ : AddCommGroup N₃
    inst✝⁶ : Module R N₃
    i₁ : LinearMap (RingHom.id R) M₁ N₁
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    i₃ : LinearMap (RingHom.id R) M₃ N₃
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    hf : Function.Exact ⇑f₁ ⇑f₂
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hg : Function.Exact ⇑g₁ ⇑g₂
    h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
    h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
    σ : M₃ → M₂
    hσ : Eq (Function.comp (⇑f₂) σ) id
    ρ : N₂ → N₁
    hρ : Eq (Function.comp ρ ⇑g₁) id
    K₂ : Type u_2
    K₃ : Type u_3
    C₁ : Type u_6
    inst✝⁵ : AddCommGroup K₂
    inst✝⁴ : Module R K₂
    inst✝³ : AddCommGroup K₃
    inst✝² : Module R K₃
    inst✝¹ : AddCommGroup C₁
    inst✝ : Module R C₁
    ι₂ : LinearMap (RingHom.id R) K₂ M₂
    hι₂ : Function.Exact ⇑ι₂ ⇑i₂
    ι₃ : LinearMap (RingHom.id R) K₃ M₃
    hι₃ : Function.Exact ⇑ι₃ ⇑i₃
    π₁ : LinearMap (RingHom.id R) N₁ C₁
    hπ₁ : Function.Exact ⇑i₁ ⇑π₁
    F : LinearMap (RingHom.id R) K₂ K₃
    hF : Eq (f₂.comp ι₂) (ι₃.comp F)
    h : Function.Injective ⇑ι₃
    H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
    H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
    x : K₃
    ⊢ Iff (Eq ((SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ ι₃ hι₃ π₁  …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝¹⁸ : CommRing R
      M₁ : Type u_7
      M₂ : Type u_5
      M₃ : Type u_4
      N₁ : Type u_8
      N₂ : Type u_9
      N₃ : Type u_10
      inst✝¹⁷ : AddCommGroup M₁
      inst✝¹⁶ : Module R M₁
      inst✝¹⁵ : AddCommGroup M₂
      inst✝¹⁴ : Module R M₂
      inst✝¹³ : AddCommGroup M₃
      inst✝¹² : Module R M₃
      inst✝¹¹ : AddCommGroup N₁
      inst✝¹⁰ : Module R N₁
      inst✝⁹ : AddCommGroup N₂
      inst✝⁸ : Module R N₂
      inst✝⁷ : AddCommGroup N₃
      inst✝⁶ : Module R N₃
      i₁ : LinearMap (RingHom.id R) M₁ N₁
      i₂ : LinearMap (RingHom.id R) M₂ N₂
      i₃ : LinearMap (RingHom.id R) M₃ N₃
      f₁ : LinearMap (RingHom.id R) M₁ M₂
      f₂ : LinearMap (RingHom.id R) M₂ M₃
      hf : Function.Exact ⇑f₁ ⇑f₂
      g₁ : LinearMap (RingHom.id R) N₁ N₂
      g₂ : LinearMap (RingHom.id R) N₂ N₃
      hg : Function.Exact ⇑g₁ ⇑g₂
      h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
      h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
      σ : M₃ → M₂
      hσ : Eq (Function.comp (⇑f₂) σ) id
      ρ : N₂ → N₁
      hρ : Eq (Function.comp ρ ⇑g₁) id
      K₂ : Type u_2
      K₃ : Type u_3
      C₁ : Type u_6
      inst✝⁵ : AddCommGroup K₂
      inst✝⁴ : Module R K₂
      inst✝³ : AddCommGroup K₃
      inst✝² : Module R K₃
      inst✝¹ : AddCommGroup C₁
      inst✝ : Module R C₁
      ι₂ : LinearMap (RingHom.id R) K₂ M₂
      hι₂ : Function.Exact ⇑ι₂ ⇑i₂
      ι₃ : LinearMap (RingHom.id R) K₃ M₃
      hι₃ : Function.Exact ⇑ι₃ ⇑i₃
      π₁ : LinearMap (RingHom.id R) N₁ C₁
      hπ₁ : Function.Exact ⇑i₁ ⇑π₁
      F : LinearMap (RingHom.id R) K₂ K₃
      hF : Eq (f₂.comp ι₂) (ι₃.comp F)
      h : Function.Injective ⇑ι₃
      H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
      H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
      x : K₃
      ⊢ Eq ((SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ ι₃ hι₃ π₁ hπ₁)  …
    -/
  · intro H
    /-
      case mp
      R : Type u_1
      inst✝¹⁸ : CommRing R
      M₁ : Type u_7
      M₂ : Type u_5
      M₃ : Type u_4
      N₁ : Type u_8
      N₂ : Type u_9
      N₃ : Type u_10
      inst✝¹⁷ : AddCommGroup M₁
      inst✝¹⁶ : Module R M₁
      inst✝¹⁵ : AddCommGroup M₂
      inst✝¹⁴ : Module R M₂
      inst✝¹³ : AddCommGroup M₃
      inst✝¹² : Module R M₃
      inst✝¹¹ : AddCommGroup N₁
      inst✝¹⁰ : Module R N₁
      inst✝⁹ : AddCommGroup N₂
      inst✝⁸ : Module R N₂
      inst✝⁷ : AddCommGroup N₃
      inst✝⁶ : Module R N₃
      i₁ : LinearMap (RingHom.id R) M₁ N₁
      i₂ : LinearMap (RingHom.id R) M₂ N₂
      i₃ : LinearMap (RingHom.id R) M₃ N₃
      f₁ : LinearMap (RingHom.id R) M₁ M₂
      f₂ : LinearMap (RingHom.id R) M₂ M₃
      hf : Function.Exact ⇑f₁ ⇑f₂
      g₁ : LinearMap (RingHom.id R) N₁ N₂
      g₂ : LinearMap (RingHom.id R) N₂ N₃
      hg : Function.Exact ⇑g₁ ⇑g₂
      h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
      h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
      σ : M₃ → M₂
      hσ : Eq (Function.comp (⇑f₂) σ) id
      ρ : N₂ → N₁
      hρ : Eq (Function.comp ρ ⇑g₁) id
      K₂ : Type u_2
      K₃ : Type u_3
      C₁ : Type u_6
      inst✝⁵ : AddCommGroup K₂
      inst✝⁴ : Module R K₂
      inst✝³ : AddCommGroup K₃
      inst✝² : Module R K₃
      inst✝¹ : AddCommGroup C₁
      inst✝ : Module R C₁
      ι₂ : LinearMap (RingHom.id R) K₂ M₂
      hι₂ : Function.Exact ⇑ι₂ ⇑i₂
      ι₃ : LinearMap (RingHom.id R) K₃ M₃
      hι₃ : Function.Exact ⇑ι₃ ⇑i₃
      π₁ : LinearMap (RingHom.id R) N₁ C₁
      hπ₁ : Function.Exact ⇑i₁ ⇑π₁
      F : LinearMap (RingHom.id R) K₂ K₃
      hF : Eq (f₂.comp ι₂) (ι₃.comp F)
      h : Function.Injective ⇑ι₃
      H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
      H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
      x : K₃
      H : Eq ((SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ ι₃ hι₃ π₁ hπ₁ …
      ⊢ Membership.mem (Set.range ⇑F) x
    -/
    obtain ⟨y, hy⟩ := (hπ₁ _).mp H
    obtain ⟨k, hk⟩ : σ (ι₃ x) - f₁ y ∈ Set.range ι₂ := by
      rw [← hι₂, map_sub, ← H₂, ← hy, sub_eq_zero]; exact congr($h₁ y)
    /-
      case mp.intro.intro
      R : Type u_1
      inst✝¹⁸ : CommRing R
      M₁ : Type u_7
      M₂ : Type u_5
      M₃ : Type u_4
      N₁ : Type u_8
      N₂ : Type u_9
      N₃ : Type u_10
      inst✝¹⁷ : AddCommGroup M₁
      inst✝¹⁶ : Module R M₁
      inst✝¹⁵ : AddCommGroup M₂
      inst✝¹⁴ : Module R M₂
      inst✝¹³ : AddCommGroup M₃
      inst✝¹² : Module R M₃
      inst✝¹¹ : AddCommGroup N₁
      inst✝¹⁰ : Module R N₁
      inst✝⁹ : AddCommGroup N₂
      inst✝⁸ : Module R N₂
      inst✝⁷ : AddCommGroup N₃
      inst✝⁶ : Module R N₃
      i₁ : LinearMap (RingHom.id R) M₁ N₁
      i₂ : LinearMap (RingHom.id R) M₂ N₂
      i₃ : LinearMap (RingHom.id R) M₃ N₃
      f₁ : LinearMap (RingHom.id R) M₁ M₂
      f₂ : LinearMap (RingHom.id R) M₂ M₃
      hf : Function.Exact ⇑f₁ ⇑f₂
      g₁ : LinearMap (RingHom.id R) N₁ N₂
      g₂ : LinearMap (RingHom.id R) N₂ N₃
      hg : Function.Exact ⇑g₁ ⇑g₂
      h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
      h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
      σ : M₃ → M₂
      hσ : Eq (Function.comp (⇑f₂) σ) id
      ρ : N₂ → N₁
      hρ : Eq (Function.comp ρ ⇑g₁) id
      K₂ : Type u_2
      K₃ : Type u_3
      C₁ : Type u_6
      inst✝⁵ : AddCommGroup K₂
      inst✝⁴ : Module R K₂
      inst✝³ : AddCommGroup K₃
      inst✝² : Module R K₃
      inst✝¹ : AddCommGroup C₁
      inst✝ : Module R C₁
      ι₂ : LinearMap (RingHom.id R) K₂ M₂
      hι₂ : Function.Exact ⇑ι₂ ⇑i₂
      ι₃ : LinearMap (RingHom.id R) K₃ M₃
      hι₃ : Function.Exact ⇑ι₃ ⇑i₃
      π₁ : LinearMap (RingHom.id R) N₁ C₁
      hπ₁ : Function.Exact ⇑i₁ ⇑π₁
      F : LinearMap (RingHom.id R) K₂ K₃
      hF : Eq (f₂.comp ι₂) (ι₃.comp F)
      h : Function.Injective ⇑ι₃
      H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
      H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
      x : K₃
      H : Eq ((SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ ι₃ hι₃ π₁ hπ₁ …
      y : M₁
      hy : Eq (i₁ y) (ρ (i₂ (σ (ι₃ x))))
      k : K₂
      hk : Eq (ι₂ k) (HSub.hSub (σ (ι₃ x)) (f₁ y))
      ⊢ Membership.mem (Set.range ⇑F) x
    -/
    refine ⟨k, h ?_⟩
    /-
      case mp.intro.intro
      R : Type u_1
      inst✝¹⁸ : CommRing R
      M₁ : Type u_7
      M₂ : Type u_5
      M₃ : Type u_4
      N₁ : Type u_8
      N₂ : Type u_9
      N₃ : Type u_10
      inst✝¹⁷ : AddCommGroup M₁
      inst✝¹⁶ : Module R M₁
      inst✝¹⁵ : AddCommGroup M₂
      inst✝¹⁴ : Module R M₂
      inst✝¹³ : AddCommGroup M₃
      inst✝¹² : Module R M₃
      inst✝¹¹ : AddCommGroup N₁
      inst✝¹⁰ : Module R N₁
      inst✝⁹ : AddCommGroup N₂
      inst✝⁸ : Module R N₂
      inst✝⁷ : AddCommGroup N₃
      inst✝⁶ : Module R N₃
      i₁ : LinearMap (RingHom.id R) M₁ N₁
      i₂ : LinearMap (RingHom.id R) M₂ N₂
      i₃ : LinearMap (RingHom.id R) M₃ N₃
      f₁ : LinearMap (RingHom.id R) M₁ M₂
      f₂ : LinearMap (RingHom.id R) M₂ M₃
      hf : Function.Exact ⇑f₁ ⇑f₂
      g₁ : LinearMap (RingHom.id R) N₁ N₂
      g₂ : LinearMap (RingHom.id R) N₂ N₃
      hg : Function.Exact ⇑g₁ ⇑g₂
      h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
      h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
      σ : M₃ → M₂
      hσ : Eq (Function.comp (⇑f₂) σ) id
      ρ : N₂ → N₁
      hρ : Eq (Function.comp ρ ⇑g₁) id
      K₂ : Type u_2
      K₃ : Type u_3
      C₁ : Type u_6
      inst✝⁵ : AddCommGroup K₂
      inst✝⁴ : Module R K₂
      inst✝³ : AddCommGroup K₃
      inst✝² : Module R K₃
      inst✝¹ : AddCommGroup C₁
      inst✝ : Module R C₁
      ι₂ : LinearMap (RingHom.id R) K₂ M₂
      hι₂ : Function.Exact ⇑ι₂ ⇑i₂
      ι₃ : LinearMap (RingHom.id R) K₃ M₃
      hι₃ : Function.Exact ⇑ι₃ ⇑i₃
      π₁ : LinearMap (RingHom.id R) N₁ C₁
      hπ₁ : Function.Exact ⇑i₁ ⇑π₁
      F : LinearMap (RingHom.id R) K₂ K₃
      hF : Eq (f₂.comp ι₂) (ι₃.comp F)
      h : Function.Injective ⇑ι₃
      H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
      H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
      x : K₃
      H : Eq ((SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ ι₃ hι₃ π₁ hπ₁ …
      y : M₁
      hy : Eq (i₁ y) (ρ (i₂ (σ (ι₃ x))))
      k : K₂
      hk : Eq (ι₂ k) (HSub.hSub (σ (ι₃ x)) (f₁ y))
      ⊢ Eq (ι₃ (F k)) (ι₃ x)
    -/
    rw [← ι₃.comp_apply, ← hF, f₂.comp_apply, hk, map_sub, H₁, hf.apply_apply_eq_zero, sub_zero]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝¹⁸ : CommRing R
      M₁ : Type u_7
      M₂ : Type u_5
      M₃ : Type u_4
      N₁ : Type u_8
      N₂ : Type u_9
      N₃ : Type u_10
      inst✝¹⁷ : AddCommGroup M₁
      inst✝¹⁶ : Module R M₁
      inst✝¹⁵ : AddCommGroup M₂
      inst✝¹⁴ : Module R M₂
      inst✝¹³ : AddCommGroup M₃
      inst✝¹² : Module R M₃
      inst✝¹¹ : AddCommGroup N₁
      inst✝¹⁰ : Module R N₁
      inst✝⁹ : AddCommGroup N₂
      inst✝⁸ : Module R N₂
      inst✝⁷ : AddCommGroup N₃
      inst✝⁶ : Module R N₃
      i₁ : LinearMap (RingHom.id R) M₁ N₁
      i₂ : LinearMap (RingHom.id R) M₂ N₂
      i₃ : LinearMap (RingHom.id R) M₃ N₃
      f₁ : LinearMap (RingHom.id R) M₁ M₂
      f₂ : LinearMap (RingHom.id R) M₂ M₃
      hf : Function.Exact ⇑f₁ ⇑f₂
      g₁ : LinearMap (RingHom.id R) N₁ N₂
      g₂ : LinearMap (RingHom.id R) N₂ N₃
      hg : Function.Exact ⇑g₁ ⇑g₂
      h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
      h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
      σ : M₃ → M₂
      hσ : Eq (Function.comp (⇑f₂) σ) id
      ρ : N₂ → N₁
      hρ : Eq (Function.comp ρ ⇑g₁) id
      K₂ : Type u_2
      K₃ : Type u_3
      C₁ : Type u_6
      inst✝⁵ : AddCommGroup K₂
      inst✝⁴ : Module R K₂
      inst✝³ : AddCommGroup K₃
      inst✝² : Module R K₃
      inst✝¹ : AddCommGroup C₁
      inst✝ : Module R C₁
      ι₂ : LinearMap (RingHom.id R) K₂ M₂
      hι₂ : Function.Exact ⇑ι₂ ⇑i₂
      ι₃ : LinearMap (RingHom.id R) K₃ M₃
      hι₃ : Function.Exact ⇑ι₃ ⇑i₃
      π₁ : LinearMap (RingHom.id R) N₁ C₁
      hπ₁ : Function.Exact ⇑i₁ ⇑π₁
      F : LinearMap (RingHom.id R) K₂ K₃
      hF : Eq (f₂.comp ι₂) (ι₃.comp F)
      h : Function.Injective ⇑ι₃
      H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
      H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
      x : K₃
      ⊢ Membership.mem (Set.range ⇑F) x → Eq ((SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂  …
    -/
  · rintro ⟨y, rfl⟩
    exact (δ_eq i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ ι₃ hι₃ π₁ hπ₁ _ (ι₂ y) congr($hF y)
      _ (by rw [map_zero, hι₂.apply_apply_eq_zero])).trans π₁.map_zero


include hπ₂ in
/--
Supppose we have an exact commutative diagram
```
                K₃
                |
                ι₃
                ↓
M₁ -f₁→ M₂ -f₂→ M₃
|       |       |
i₁      i₂      i₃
↓       ↓       ↓
N₁ -g₁→ N₂ -g₂→ N₃
|       |
π₁      π₂
↓       ↓
C₁ -G-→ C₂

```
such that `f₂` is surjective with a (set-theoretic) section `σ`, `g₁` is injective with a
(set-theoretic) retraction `ρ`, and `π₁` is surjective, then `K₂ -δ→ C₁ -G→ C₂` is exact.
-/
lemma SnakeLemma.exact_δ_left (G : C₁ →ₗ[R] C₂) (hF : G.comp π₁ = π₂.comp g₁) (h : Surjective π₁) :
    Exact (δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ ι₃ hι₃ π₁ hπ₁) G := by
  /-
    R : Type u_1
    inst✝¹⁸ : CommRing R
    M₁ : Type u_7
    M₂ : Type u_8
    M₃ : Type u_9
    N₁ : Type u_4
    N₂ : Type u_5
    N₃ : Type u_10
    inst✝¹⁷ : AddCommGroup M₁
    inst✝¹⁶ : Module R M₁
    inst✝¹⁵ : AddCommGroup M₂
    inst✝¹⁴ : Module R M₂
    inst✝¹³ : AddCommGroup M₃
    inst✝¹² : Module R M₃
    inst✝¹¹ : AddCommGroup N₁
    inst✝¹⁰ : Module R N₁
    inst✝⁹ : AddCommGroup N₂
    inst✝⁸ : Module R N₂
    inst✝⁷ : AddCommGroup N₃
    inst✝⁶ : Module R N₃
    i₁ : LinearMap (RingHom.id R) M₁ N₁
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    i₃ : LinearMap (RingHom.id R) M₃ N₃
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    hf : Function.Exact ⇑f₁ ⇑f₂
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hg : Function.Exact ⇑g₁ ⇑g₂
    h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
    h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
    σ : M₃ → M₂
    hσ : Eq (Function.comp (⇑f₂) σ) id
    ρ : N₂ → N₁
    hρ : Eq (Function.comp ρ ⇑g₁) id
    K₃ : Type u_6
    C₁ : Type u_2
    C₂ : Type u_3
    inst✝⁵ : AddCommGroup K₃
    inst✝⁴ : Module R K₃
    inst✝³ : AddCommGroup C₁
    inst✝² : Module R C₁
    inst✝¹ : AddCommGroup C₂
    inst✝ : Module R C₂
    ι₃ : LinearMap (RingHom.id R) K₃ M₃
    hι₃ : Function.Exact ⇑ι₃ ⇑i₃
    π₁ : LinearMap (RingHom.id R) N₁ C₁
    hπ₁ : Function.Exact ⇑i₁ ⇑π₁
    π₂ : LinearMap (RingHom.id R) N₂ C₂
    hπ₂ : Function.Exact ⇑i₂ ⇑π₂
    G : LinearMap (RingHom.id R) C₁ C₂
    hF : Eq (G.comp π₁) (π₂.comp g₁)
    h : Function.Surjective ⇑π₁
    ⊢ Function.Exact ⇑(SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ ι₃  …
  -/
  haveI H₁ : ∀ x, f₂ (σ x) = x := congr_fun hσ
  /-
    R : Type u_1
    inst✝¹⁸ : CommRing R
    M₁ : Type u_7
    M₂ : Type u_8
    M₃ : Type u_9
    N₁ : Type u_4
    N₂ : Type u_5
    N₃ : Type u_10
    inst✝¹⁷ : AddCommGroup M₁
    inst✝¹⁶ : Module R M₁
    inst✝¹⁵ : AddCommGroup M₂
    inst✝¹⁴ : Module R M₂
    inst✝¹³ : AddCommGroup M₃
    inst✝¹² : Module R M₃
    inst✝¹¹ : AddCommGroup N₁
    inst✝¹⁰ : Module R N₁
    inst✝⁹ : AddCommGroup N₂
    inst✝⁸ : Module R N₂
    inst✝⁷ : AddCommGroup N₃
    inst✝⁶ : Module R N₃
    i₁ : LinearMap (RingHom.id R) M₁ N₁
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    i₃ : LinearMap (RingHom.id R) M₃ N₃
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    hf : Function.Exact ⇑f₁ ⇑f₂
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hg : Function.Exact ⇑g₁ ⇑g₂
    h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
    h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
    σ : M₃ → M₂
    hσ : Eq (Function.comp (⇑f₂) σ) id
    ρ : N₂ → N₁
    hρ : Eq (Function.comp ρ ⇑g₁) id
    K₃ : Type u_6
    C₁ : Type u_2
    C₂ : Type u_3
    inst✝⁵ : AddCommGroup K₃
    inst✝⁴ : Module R K₃
    inst✝³ : AddCommGroup C₁
    inst✝² : Module R C₁
    inst✝¹ : AddCommGroup C₂
    inst✝ : Module R C₂
    ι₃ : LinearMap (RingHom.id R) K₃ M₃
    hι₃ : Function.Exact ⇑ι₃ ⇑i₃
    π₁ : LinearMap (RingHom.id R) N₁ C₁
    hπ₁ : Function.Exact ⇑i₁ ⇑π₁
    π₂ : LinearMap (RingHom.id R) N₂ C₂
    hπ₂ : Function.Exact ⇑i₂ ⇑π₂
    G : LinearMap (RingHom.id R) C₁ C₂
    hF : Eq (G.comp π₁) (π₂.comp g₁)
    h : Function.Surjective ⇑π₁
    H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
    ⊢ Function.Exact ⇑(SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ ι₃  …
  -/
  haveI H₂ := δ_aux i₂ i₃ f₂ g₁ g₂ hg h₂ σ hσ ρ hρ ι₃ hι₃
  /-
    R : Type u_1
    inst✝¹⁸ : CommRing R
    M₁ : Type u_7
    M₂ : Type u_8
    M₃ : Type u_9
    N₁ : Type u_4
    N₂ : Type u_5
    N₃ : Type u_10
    inst✝¹⁷ : AddCommGroup M₁
    inst✝¹⁶ : Module R M₁
    inst✝¹⁵ : AddCommGroup M₂
    inst✝¹⁴ : Module R M₂
    inst✝¹³ : AddCommGroup M₃
    inst✝¹² : Module R M₃
    inst✝¹¹ : AddCommGroup N₁
    inst✝¹⁰ : Module R N₁
    inst✝⁹ : AddCommGroup N₂
    inst✝⁸ : Module R N₂
    inst✝⁷ : AddCommGroup N₃
    inst✝⁶ : Module R N₃
    i₁ : LinearMap (RingHom.id R) M₁ N₁
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    i₃ : LinearMap (RingHom.id R) M₃ N₃
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    hf : Function.Exact ⇑f₁ ⇑f₂
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hg : Function.Exact ⇑g₁ ⇑g₂
    h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
    h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
    σ : M₃ → M₂
    hσ : Eq (Function.comp (⇑f₂) σ) id
    ρ : N₂ → N₁
    hρ : Eq (Function.comp ρ ⇑g₁) id
    K₃ : Type u_6
    C₁ : Type u_2
    C₂ : Type u_3
    inst✝⁵ : AddCommGroup K₃
    inst✝⁴ : Module R K₃
    inst✝³ : AddCommGroup C₁
    inst✝² : Module R C₁
    inst✝¹ : AddCommGroup C₂
    inst✝ : Module R C₂
    ι₃ : LinearMap (RingHom.id R) K₃ M₃
    hι₃ : Function.Exact ⇑ι₃ ⇑i₃
    π₁ : LinearMap (RingHom.id R) N₁ C₁
    hπ₁ : Function.Exact ⇑i₁ ⇑π₁
    π₂ : LinearMap (RingHom.id R) N₂ C₂
    hπ₂ : Function.Exact ⇑i₂ ⇑π₂
    G : LinearMap (RingHom.id R) C₁ C₂
    hF : Eq (G.comp π₁) (π₂.comp g₁)
    h : Function.Surjective ⇑π₁
    H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
    H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
    ⊢ Function.Exact ⇑(SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ ι₃  …
  -/
  intro x
  /-
    R : Type u_1
    inst✝¹⁸ : CommRing R
    M₁ : Type u_7
    M₂ : Type u_8
    M₃ : Type u_9
    N₁ : Type u_4
    N₂ : Type u_5
    N₃ : Type u_10
    inst✝¹⁷ : AddCommGroup M₁
    inst✝¹⁶ : Module R M₁
    inst✝¹⁵ : AddCommGroup M₂
    inst✝¹⁴ : Module R M₂
    inst✝¹³ : AddCommGroup M₃
    inst✝¹² : Module R M₃
    inst✝¹¹ : AddCommGroup N₁
    inst✝¹⁰ : Module R N₁
    inst✝⁹ : AddCommGroup N₂
    inst✝⁸ : Module R N₂
    inst✝⁷ : AddCommGroup N₃
    inst✝⁶ : Module R N₃
    i₁ : LinearMap (RingHom.id R) M₁ N₁
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    i₃ : LinearMap (RingHom.id R) M₃ N₃
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    hf : Function.Exact ⇑f₁ ⇑f₂
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hg : Function.Exact ⇑g₁ ⇑g₂
    h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
    h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
    σ : M₃ → M₂
    hσ : Eq (Function.comp (⇑f₂) σ) id
    ρ : N₂ → N₁
    hρ : Eq (Function.comp ρ ⇑g₁) id
    K₃ : Type u_6
    C₁ : Type u_2
    C₂ : Type u_3
    inst✝⁵ : AddCommGroup K₃
    inst✝⁴ : Module R K₃
    inst✝³ : AddCommGroup C₁
    inst✝² : Module R C₁
    inst✝¹ : AddCommGroup C₂
    inst✝ : Module R C₂
    ι₃ : LinearMap (RingHom.id R) K₃ M₃
    hι₃ : Function.Exact ⇑ι₃ ⇑i₃
    π₁ : LinearMap (RingHom.id R) N₁ C₁
    hπ₁ : Function.Exact ⇑i₁ ⇑π₁
    π₂ : LinearMap (RingHom.id R) N₂ C₂
    hπ₂ : Function.Exact ⇑i₂ ⇑π₂
    G : LinearMap (RingHom.id R) C₁ C₂
    hF : Eq (G.comp π₁) (π₂.comp g₁)
    h : Function.Surjective ⇑π₁
    H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
    H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
    x : C₁
    ⊢ Iff (Eq (G x) 0) (Membership.mem (Set.range ⇑(SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝¹⁸ : CommRing R
      M₁ : Type u_7
      M₂ : Type u_8
      M₃ : Type u_9
      N₁ : Type u_4
      N₂ : Type u_5
      N₃ : Type u_10
      inst✝¹⁷ : AddCommGroup M₁
      inst✝¹⁶ : Module R M₁
      inst✝¹⁵ : AddCommGroup M₂
      inst✝¹⁴ : Module R M₂
      inst✝¹³ : AddCommGroup M₃
      inst✝¹² : Module R M₃
      inst✝¹¹ : AddCommGroup N₁
      inst✝¹⁰ : Module R N₁
      inst✝⁹ : AddCommGroup N₂
      inst✝⁸ : Module R N₂
      inst✝⁷ : AddCommGroup N₃
      inst✝⁶ : Module R N₃
      i₁ : LinearMap (RingHom.id R) M₁ N₁
      i₂ : LinearMap (RingHom.id R) M₂ N₂
      i₃ : LinearMap (RingHom.id R) M₃ N₃
      f₁ : LinearMap (RingHom.id R) M₁ M₂
      f₂ : LinearMap (RingHom.id R) M₂ M₃
      hf : Function.Exact ⇑f₁ ⇑f₂
      g₁ : LinearMap (RingHom.id R) N₁ N₂
      g₂ : LinearMap (RingHom.id R) N₂ N₃
      hg : Function.Exact ⇑g₁ ⇑g₂
      h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
      h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
      σ : M₃ → M₂
      hσ : Eq (Function.comp (⇑f₂) σ) id
      ρ : N₂ → N₁
      hρ : Eq (Function.comp ρ ⇑g₁) id
      K₃ : Type u_6
      C₁ : Type u_2
      C₂ : Type u_3
      inst✝⁵ : AddCommGroup K₃
      inst✝⁴ : Module R K₃
      inst✝³ : AddCommGroup C₁
      inst✝² : Module R C₁
      inst✝¹ : AddCommGroup C₂
      inst✝ : Module R C₂
      ι₃ : LinearMap (RingHom.id R) K₃ M₃
      hι₃ : Function.Exact ⇑ι₃ ⇑i₃
      π₁ : LinearMap (RingHom.id R) N₁ C₁
      hπ₁ : Function.Exact ⇑i₁ ⇑π₁
      π₂ : LinearMap (RingHom.id R) N₂ C₂
      hπ₂ : Function.Exact ⇑i₂ ⇑π₂
      G : LinearMap (RingHom.id R) C₁ C₂
      hF : Eq (G.comp π₁) (π₂.comp g₁)
      h : Function.Surjective ⇑π₁
      H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
      H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
      x : C₁
      ⊢ Eq (G x) 0 → Membership.mem (Set.range ⇑(SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g …
    -/
  · intro H
    /-
      case mp
      R : Type u_1
      inst✝¹⁸ : CommRing R
      M₁ : Type u_7
      M₂ : Type u_8
      M₃ : Type u_9
      N₁ : Type u_4
      N₂ : Type u_5
      N₃ : Type u_10
      inst✝¹⁷ : AddCommGroup M₁
      inst✝¹⁶ : Module R M₁
      inst✝¹⁵ : AddCommGroup M₂
      inst✝¹⁴ : Module R M₂
      inst✝¹³ : AddCommGroup M₃
      inst✝¹² : Module R M₃
      inst✝¹¹ : AddCommGroup N₁
      inst✝¹⁰ : Module R N₁
      inst✝⁹ : AddCommGroup N₂
      inst✝⁸ : Module R N₂
      inst✝⁷ : AddCommGroup N₃
      inst✝⁶ : Module R N₃
      i₁ : LinearMap (RingHom.id R) M₁ N₁
      i₂ : LinearMap (RingHom.id R) M₂ N₂
      i₃ : LinearMap (RingHom.id R) M₃ N₃
      f₁ : LinearMap (RingHom.id R) M₁ M₂
      f₂ : LinearMap (RingHom.id R) M₂ M₃
      hf : Function.Exact ⇑f₁ ⇑f₂
      g₁ : LinearMap (RingHom.id R) N₁ N₂
      g₂ : LinearMap (RingHom.id R) N₂ N₃
      hg : Function.Exact ⇑g₁ ⇑g₂
      h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
      h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
      σ : M₃ → M₂
      hσ : Eq (Function.comp (⇑f₂) σ) id
      ρ : N₂ → N₁
      hρ : Eq (Function.comp ρ ⇑g₁) id
      K₃ : Type u_6
      C₁ : Type u_2
      C₂ : Type u_3
      inst✝⁵ : AddCommGroup K₃
      inst✝⁴ : Module R K₃
      inst✝³ : AddCommGroup C₁
      inst✝² : Module R C₁
      inst✝¹ : AddCommGroup C₂
      inst✝ : Module R C₂
      ι₃ : LinearMap (RingHom.id R) K₃ M₃
      hι₃ : Function.Exact ⇑ι₃ ⇑i₃
      π₁ : LinearMap (RingHom.id R) N₁ C₁
      hπ₁ : Function.Exact ⇑i₁ ⇑π₁
      π₂ : LinearMap (RingHom.id R) N₂ C₂
      hπ₂ : Function.Exact ⇑i₂ ⇑π₂
      G : LinearMap (RingHom.id R) C₁ C₂
      hF : Eq (G.comp π₁) (π₂.comp g₁)
      h : Function.Surjective ⇑π₁
      H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
      H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
      x : C₁
      H : Eq (G x) 0
      ⊢ Membership.mem (Set.range ⇑(SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ  …
    -/
    obtain ⟨x, rfl⟩ := h x
    /-
      case mp.intro
      R : Type u_1
      inst✝¹⁸ : CommRing R
      M₁ : Type u_7
      M₂ : Type u_8
      M₃ : Type u_9
      N₁ : Type u_4
      N₂ : Type u_5
      N₃ : Type u_10
      inst✝¹⁷ : AddCommGroup M₁
      inst✝¹⁶ : Module R M₁
      inst✝¹⁵ : AddCommGroup M₂
      inst✝¹⁴ : Module R M₂
      inst✝¹³ : AddCommGroup M₃
      inst✝¹² : Module R M₃
      inst✝¹¹ : AddCommGroup N₁
      inst✝¹⁰ : Module R N₁
      inst✝⁹ : AddCommGroup N₂
      inst✝⁸ : Module R N₂
      inst✝⁷ : AddCommGroup N₃
      inst✝⁶ : Module R N₃
      i₁ : LinearMap (RingHom.id R) M₁ N₁
      i₂ : LinearMap (RingHom.id R) M₂ N₂
      i₃ : LinearMap (RingHom.id R) M₃ N₃
      f₁ : LinearMap (RingHom.id R) M₁ M₂
      f₂ : LinearMap (RingHom.id R) M₂ M₃
      hf : Function.Exact ⇑f₁ ⇑f₂
      g₁ : LinearMap (RingHom.id R) N₁ N₂
      g₂ : LinearMap (RingHom.id R) N₂ N₃
      hg : Function.Exact ⇑g₁ ⇑g₂
      h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
      h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
      σ : M₃ → M₂
      hσ : Eq (Function.comp (⇑f₂) σ) id
      ρ : N₂ → N₁
      hρ : Eq (Function.comp ρ ⇑g₁) id
      K₃ : Type u_6
      C₁ : Type u_2
      C₂ : Type u_3
      inst✝⁵ : AddCommGroup K₃
      inst✝⁴ : Module R K₃
      inst✝³ : AddCommGroup C₁
      inst✝² : Module R C₁
      inst✝¹ : AddCommGroup C₂
      inst✝ : Module R C₂
      ι₃ : LinearMap (RingHom.id R) K₃ M₃
      hι₃ : Function.Exact ⇑ι₃ ⇑i₃
      π₁ : LinearMap (RingHom.id R) N₁ C₁
      hπ₁ : Function.Exact ⇑i₁ ⇑π₁
      π₂ : LinearMap (RingHom.id R) N₂ C₂
      hπ₂ : Function.Exact ⇑i₂ ⇑π₂
      G : LinearMap (RingHom.id R) C₁ C₂
      hF : Eq (G.comp π₁) (π₂.comp g₁)
      h : Function.Surjective ⇑π₁
      H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
      H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
      x : N₁
      H : Eq (G (π₁ x)) 0
      ⊢ Membership.mem (Set.range ⇑(SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ  …
    -/
    obtain ⟨y, hy⟩ := (hπ₂ (g₁ x)).mp (by simpa only [← LinearMap.comp_apply, hF] using H)
    obtain ⟨z, hz⟩ : f₂ y ∈ range ι₃ := (hι₃ (f₂ y)).mp (by rw [← i₃.comp_apply, ← h₂,
      g₂.comp_apply, hy, hg.apply_apply_eq_zero])
    /-
      case mp.intro.intro.intro
      R : Type u_1
      inst✝¹⁸ : CommRing R
      M₁ : Type u_7
      M₂ : Type u_8
      M₃ : Type u_9
      N₁ : Type u_4
      N₂ : Type u_5
      N₃ : Type u_10
      inst✝¹⁷ : AddCommGroup M₁
      inst✝¹⁶ : Module R M₁
      inst✝¹⁵ : AddCommGroup M₂
      inst✝¹⁴ : Module R M₂
      inst✝¹³ : AddCommGroup M₃
      inst✝¹² : Module R M₃
      inst✝¹¹ : AddCommGroup N₁
      inst✝¹⁰ : Module R N₁
      inst✝⁹ : AddCommGroup N₂
      inst✝⁸ : Module R N₂
      inst✝⁷ : AddCommGroup N₃
      inst✝⁶ : Module R N₃
      i₁ : LinearMap (RingHom.id R) M₁ N₁
      i₂ : LinearMap (RingHom.id R) M₂ N₂
      i₃ : LinearMap (RingHom.id R) M₃ N₃
      f₁ : LinearMap (RingHom.id R) M₁ M₂
      f₂ : LinearMap (RingHom.id R) M₂ M₃
      hf : Function.Exact ⇑f₁ ⇑f₂
      g₁ : LinearMap (RingHom.id R) N₁ N₂
      g₂ : LinearMap (RingHom.id R) N₂ N₃
      hg : Function.Exact ⇑g₁ ⇑g₂
      h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
      h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
      σ : M₃ → M₂
      hσ : Eq (Function.comp (⇑f₂) σ) id
      ρ : N₂ → N₁
      hρ : Eq (Function.comp ρ ⇑g₁) id
      K₃ : Type u_6
      C₁ : Type u_2
      C₂ : Type u_3
      inst✝⁵ : AddCommGroup K₃
      inst✝⁴ : Module R K₃
      inst✝³ : AddCommGroup C₁
      inst✝² : Module R C₁
      inst✝¹ : AddCommGroup C₂
      inst✝ : Module R C₂
      ι₃ : LinearMap (RingHom.id R) K₃ M₃
      hι₃ : Function.Exact ⇑ι₃ ⇑i₃
      π₁ : LinearMap (RingHom.id R) N₁ C₁
      hπ₁ : Function.Exact ⇑i₁ ⇑π₁
      π₂ : LinearMap (RingHom.id R) N₂ C₂
      hπ₂ : Function.Exact ⇑i₂ ⇑π₂
      G : LinearMap (RingHom.id R) C₁ C₂
      hF : Eq (G.comp π₁) (π₂.comp g₁)
      h : Function.Surjective ⇑π₁
      H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
      H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
      x : N₁
      H : Eq (G (π₁ x)) 0
      y : M₂
      hy : Eq (i₂ y) (g₁ x)
      z : K₃
      hz : Eq (ι₃ z) (f₂ y)
      ⊢ Membership.mem (Set.range ⇑(SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ  …
    -/
    exact ⟨z, δ_eq i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ ι₃ hι₃ π₁ hπ₁ _ _ hz.symm _ hy.symm⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝¹⁸ : CommRing R
      M₁ : Type u_7
      M₂ : Type u_8
      M₃ : Type u_9
      N₁ : Type u_4
      N₂ : Type u_5
      N₃ : Type u_10
      inst✝¹⁷ : AddCommGroup M₁
      inst✝¹⁶ : Module R M₁
      inst✝¹⁵ : AddCommGroup M₂
      inst✝¹⁴ : Module R M₂
      inst✝¹³ : AddCommGroup M₃
      inst✝¹² : Module R M₃
      inst✝¹¹ : AddCommGroup N₁
      inst✝¹⁰ : Module R N₁
      inst✝⁹ : AddCommGroup N₂
      inst✝⁸ : Module R N₂
      inst✝⁷ : AddCommGroup N₃
      inst✝⁶ : Module R N₃
      i₁ : LinearMap (RingHom.id R) M₁ N₁
      i₂ : LinearMap (RingHom.id R) M₂ N₂
      i₃ : LinearMap (RingHom.id R) M₃ N₃
      f₁ : LinearMap (RingHom.id R) M₁ M₂
      f₂ : LinearMap (RingHom.id R) M₂ M₃
      hf : Function.Exact ⇑f₁ ⇑f₂
      g₁ : LinearMap (RingHom.id R) N₁ N₂
      g₂ : LinearMap (RingHom.id R) N₂ N₃
      hg : Function.Exact ⇑g₁ ⇑g₂
      h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
      h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
      σ : M₃ → M₂
      hσ : Eq (Function.comp (⇑f₂) σ) id
      ρ : N₂ → N₁
      hρ : Eq (Function.comp ρ ⇑g₁) id
      K₃ : Type u_6
      C₁ : Type u_2
      C₂ : Type u_3
      inst✝⁵ : AddCommGroup K₃
      inst✝⁴ : Module R K₃
      inst✝³ : AddCommGroup C₁
      inst✝² : Module R C₁
      inst✝¹ : AddCommGroup C₂
      inst✝ : Module R C₂
      ι₃ : LinearMap (RingHom.id R) K₃ M₃
      hι₃ : Function.Exact ⇑ι₃ ⇑i₃
      π₁ : LinearMap (RingHom.id R) N₁ C₁
      hπ₁ : Function.Exact ⇑i₁ ⇑π₁
      π₂ : LinearMap (RingHom.id R) N₂ C₂
      hπ₂ : Function.Exact ⇑i₂ ⇑π₂
      G : LinearMap (RingHom.id R) C₁ C₂
      hF : Eq (G.comp π₁) (π₂.comp g₁)
      h : Function.Surjective ⇑π₁
      H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
      H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
      x : C₁
      ⊢ Membership.mem (Set.range ⇑(SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ  …
    -/
  · rintro ⟨x, rfl⟩
    /-
      case mpr.intro
      R : Type u_1
      inst✝¹⁸ : CommRing R
      M₁ : Type u_7
      M₂ : Type u_8
      M₃ : Type u_9
      N₁ : Type u_4
      N₂ : Type u_5
      N₃ : Type u_10
      inst✝¹⁷ : AddCommGroup M₁
      inst✝¹⁶ : Module R M₁
      inst✝¹⁵ : AddCommGroup M₂
      inst✝¹⁴ : Module R M₂
      inst✝¹³ : AddCommGroup M₃
      inst✝¹² : Module R M₃
      inst✝¹¹ : AddCommGroup N₁
      inst✝¹⁰ : Module R N₁
      inst✝⁹ : AddCommGroup N₂
      inst✝⁸ : Module R N₂
      inst✝⁷ : AddCommGroup N₃
      inst✝⁶ : Module R N₃
      i₁ : LinearMap (RingHom.id R) M₁ N₁
      i₂ : LinearMap (RingHom.id R) M₂ N₂
      i₃ : LinearMap (RingHom.id R) M₃ N₃
      f₁ : LinearMap (RingHom.id R) M₁ M₂
      f₂ : LinearMap (RingHom.id R) M₂ M₃
      hf : Function.Exact ⇑f₁ ⇑f₂
      g₁ : LinearMap (RingHom.id R) N₁ N₂
      g₂ : LinearMap (RingHom.id R) N₂ N₃
      hg : Function.Exact ⇑g₁ ⇑g₂
      h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
      h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
      σ : M₃ → M₂
      hσ : Eq (Function.comp (⇑f₂) σ) id
      ρ : N₂ → N₁
      hρ : Eq (Function.comp ρ ⇑g₁) id
      K₃ : Type u_6
      C₁ : Type u_2
      C₂ : Type u_3
      inst✝⁵ : AddCommGroup K₃
      inst✝⁴ : Module R K₃
      inst✝³ : AddCommGroup C₁
      inst✝² : Module R C₁
      inst✝¹ : AddCommGroup C₂
      inst✝ : Module R C₂
      ι₃ : LinearMap (RingHom.id R) K₃ M₃
      hι₃ : Function.Exact ⇑ι₃ ⇑i₃
      π₁ : LinearMap (RingHom.id R) N₁ C₁
      hπ₁ : Function.Exact ⇑i₁ ⇑π₁
      π₂ : LinearMap (RingHom.id R) N₂ C₂
      hπ₂ : Function.Exact ⇑i₂ ⇑π₂
      G : LinearMap (RingHom.id R) C₁ C₂
      hF : Eq (G.comp π₁) (π₂.comp g₁)
      h : Function.Surjective ⇑π₁
      H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
      H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
      x : K₃
      ⊢ Eq (G ((SnakeLemma.δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ σ hσ ρ hρ ι₃ hι₃ π₁ hπ …
    -/
    simp only [δ, id_eq, coe_mk, AddHom.coe_mk]
    /-
      case mpr.intro
      R : Type u_1
      inst✝¹⁸ : CommRing R
      M₁ : Type u_7
      M₂ : Type u_8
      M₃ : Type u_9
      N₁ : Type u_4
      N₂ : Type u_5
      N₃ : Type u_10
      inst✝¹⁷ : AddCommGroup M₁
      inst✝¹⁶ : Module R M₁
      inst✝¹⁵ : AddCommGroup M₂
      inst✝¹⁴ : Module R M₂
      inst✝¹³ : AddCommGroup M₃
      inst✝¹² : Module R M₃
      inst✝¹¹ : AddCommGroup N₁
      inst✝¹⁰ : Module R N₁
      inst✝⁹ : AddCommGroup N₂
      inst✝⁸ : Module R N₂
      inst✝⁷ : AddCommGroup N₃
      inst✝⁶ : Module R N₃
      i₁ : LinearMap (RingHom.id R) M₁ N₁
      i₂ : LinearMap (RingHom.id R) M₂ N₂
      i₃ : LinearMap (RingHom.id R) M₃ N₃
      f₁ : LinearMap (RingHom.id R) M₁ M₂
      f₂ : LinearMap (RingHom.id R) M₂ M₃
      hf : Function.Exact ⇑f₁ ⇑f₂
      g₁ : LinearMap (RingHom.id R) N₁ N₂
      g₂ : LinearMap (RingHom.id R) N₂ N₃
      hg : Function.Exact ⇑g₁ ⇑g₂
      h₁ : Eq (g₁.comp i₁) (i₂.comp f₁)
      h₂ : Eq (g₂.comp i₂) (i₃.comp f₂)
      σ : M₃ → M₂
      hσ : Eq (Function.comp (⇑f₂) σ) id
      ρ : N₂ → N₁
      hρ : Eq (Function.comp ρ ⇑g₁) id
      K₃ : Type u_6
      C₁ : Type u_2
      C₂ : Type u_3
      inst✝⁵ : AddCommGroup K₃
      inst✝⁴ : Module R K₃
      inst✝³ : AddCommGroup C₁
      inst✝² : Module R C₁
      inst✝¹ : AddCommGroup C₂
      inst✝ : Module R C₂
      ι₃ : LinearMap (RingHom.id R) K₃ M₃
      hι₃ : Function.Exact ⇑ι₃ ⇑i₃
      π₁ : LinearMap (RingHom.id R) N₁ C₁
      hπ₁ : Function.Exact ⇑i₁ ⇑π₁
      π₂ : LinearMap (RingHom.id R) N₂ C₂
      hπ₂ : Function.Exact ⇑i₂ ⇑π₂
      G : LinearMap (RingHom.id R) C₁ C₂
      hF : Eq (G.comp π₁) (π₂.comp g₁)
      h : Function.Surjective ⇑π₁
      H₁ : ∀ (x : M₃), Eq (f₂ (σ x)) x
      H₂ : ∀ (x : K₃), Eq (g₁ (ρ (i₂ (σ (ι₃ x))))) (i₂ (σ (ι₃ x)))
      x : K₃
      ⊢ Eq (G (π₁ (ρ (i₂ (σ (ι₃ x)))))) 0
    -/
    rw [← G.comp_apply, hF, π₂.comp_apply, H₂, hπ₂.apply_apply_eq_zero]
    /-
      🎉 no goals
    -/


/--
Supppose we have an exact commutative diagram
```
                K₃
                |
                ι₃
                ↓
M₁ -f₁→ M₂ -f₂→ M₃
|       |       |
i₁      i₂      i₃
↓       ↓       ↓
N₁ -g₁→ N₂ -g₂→ N₃
|
π₁
↓
C₁

```
such that `f₂` is surjective and `g₁` is injective,
then this is the linear map `K₃ → C₁` given by the snake lemma.

Also see `SnakeLemma.δ` for a computable version.
-/
noncomputable def SnakeLemma.δ' (hf₂ : Surjective f₂) (hg₁ : Injective g₁) : K₃ →ₗ[R] C₁ :=
  δ i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ _ (funext (surjInv_eq hf₂)) _ (invFun_comp hg₁) ι₃ hι₃ π₁ hπ₁


lemma SnakeLemma.δ'_eq (hf₂ : Surjective f₂) (hg₁ : Injective g₁)
    (x : K₃) (y) (hy : f₂ y = ι₃ x) (z) (hz : g₁ z = i₂ y) :
    δ' i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ ι₃ hι₃ π₁ hπ₁ hf₂ hg₁ x = π₁ z :=
  SnakeLemma.δ_eq _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ‹_› ‹_› _ ‹_›


include hι₂ in
/--
Supppose we have an exact commutative diagram
```
        K₂ -F-→ K₃
        |       |
        ι₂      ι₃
        ↓       ↓
M₁ -f₁→ M₂ -f₂→ M₃
|       |       |
i₁      i₂      i₃
↓       ↓       ↓
N₁ -g₁→ N₂ -g₂→ N₃
|
π₁
↓
C₁

```
such that `f₂` is surjective, `g₁` is injective, and `ι₃` is injective,
then `K₂ -F→ K₂ -δ→ C₁` is exact.
-/
lemma SnakeLemma.exact_δ'_right (hf₂ : Surjective f₂) (hg₁ : Injective g₁)
    (F : K₂ →ₗ[R] K₃) (hF : f₂.comp ι₂ = ι₃.comp F) (h : Injective ι₃) :
    Exact F (δ' i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ ι₃ hι₃ π₁ hπ₁ hf₂ hg₁) :=
  SnakeLemma.exact_δ_right _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ‹_› _ _ _ _ _ ‹_› ‹_›


include hπ₂ in
/--
Supppose we have an exact commutative diagram
```
                K₃
                |
                ι₃
                ↓
M₁ -f₁→ M₂ -f₂→ M₃
|       |       |
i₁      i₂      i₃
↓       ↓       ↓
N₁ -g₁→ N₂ -g₂→ N₃
|       |
π₁      π₂
↓       ↓
C₁ -G-→ C₂

```
such that `f₂` is surjective, `g₁` is injective, and `π₁` is surjective,
then `K₂ -δ→ C₁ -G→ C₂` is exact.
-/
lemma SnakeLemma.exact_δ'_left (hf₂ : Surjective f₂) (hg₁ : Injective g₁)
    (G : C₁ →ₗ[R] C₂) (hF : G.comp π₁ = π₂.comp g₁) (h : Surjective π₁) :
    Exact (δ' i₁ i₂ i₃ f₁ f₂ hf g₁ g₂ hg h₁ h₂ ι₃ hι₃ π₁ hπ₁ hf₂ hg₁) G :=
  SnakeLemma.exact_δ_left _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ‹_› _ ‹_› ‹_›

