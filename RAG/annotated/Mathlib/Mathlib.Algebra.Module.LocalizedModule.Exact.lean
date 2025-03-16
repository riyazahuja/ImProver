/-- Localization of modules is an exact functor, proven here for `LocalizedModule`.
See `IsLocalizedModule.map_exact` for the more general version. -/
lemma LocalizedModule.map_exact (g : M₀ →ₗ[R] M₁) (h : M₁ →ₗ[R] M₂) (ex : Exact g h) :
    Exact (map S (mkLinearMap S M₀) (mkLinearMap S M₁) g)
    (map S (mkLinearMap S M₁) (mkLinearMap S M₂) h) :=
  fun y ↦ Iff.intro
    (induction_on
      (fun m s hy ↦ by
        /-
          R : Type u_1
          inst✝⁶ : CommSemiring R
          S : Submonoid R
          M₀ : Type u_2
          inst✝⁵ : AddCommMonoid M₀
          inst✝⁴ : Module R M₀
          M₁ : Type u_3
          inst✝³ : AddCommMonoid M₁
          inst✝² : Module R M₁
          M₂ : Type u_4
          inst✝¹ : AddCommMonoid M₂
          inst✝ : Module R M₂
          g : LinearMap (RingHom.id R) M₀ M₁
          h : LinearMap (RingHom.id R) M₁ M₂
          ex : Function.Exact ⇑g ⇑h
          y : LocalizedModule S M₁
          m : M₁
          s : Subtype fun x => Membership.mem S x
          hy : Eq (((IsLocalizedModule.map S (LocalizedModule.mkLinearMap S M₁) (Localiz …
          ⊢ Membership.mem (Set.range ⇑((IsLocalizedModule.map S (LocalizedModule.mkLine …
        -/
        rw [map_LocalizedModules, ← zero_mk 1, mk_eq, one_smul, smul_zero] at hy
        /-
          R : Type u_1
          inst✝⁶ : CommSemiring R
          S : Submonoid R
          M₀ : Type u_2
          inst✝⁵ : AddCommMonoid M₀
          inst✝⁴ : Module R M₀
          M₁ : Type u_3
          inst✝³ : AddCommMonoid M₁
          inst✝² : Module R M₁
          M₂ : Type u_4
          inst✝¹ : AddCommMonoid M₂
          inst✝ : Module R M₂
          g : LinearMap (RingHom.id R) M₀ M₁
          h : LinearMap (RingHom.id R) M₁ M₂
          ex : Function.Exact ⇑g ⇑h
          y : LocalizedModule S M₁
          m : M₁
          s : Subtype fun x => Membership.mem S x
          hy : Exists fun u => Eq (HSMul.hSMul u (h m)) (HSMul.hSMul u 0)
          ⊢ Membership.mem (Set.range ⇑((IsLocalizedModule.map S (LocalizedModule.mkLine …
        -/
        obtain ⟨a, aS, ha⟩ := Subtype.exists.1 hy
        /-
          case intro.intro
          R : Type u_1
          inst✝⁶ : CommSemiring R
          S : Submonoid R
          M₀ : Type u_2
          inst✝⁵ : AddCommMonoid M₀
          inst✝⁴ : Module R M₀
          M₁ : Type u_3
          inst✝³ : AddCommMonoid M₁
          inst✝² : Module R M₁
          M₂ : Type u_4
          inst✝¹ : AddCommMonoid M₂
          inst✝ : Module R M₂
          g : LinearMap (RingHom.id R) M₀ M₁
          h : LinearMap (RingHom.id R) M₁ M₂
          ex : Function.Exact ⇑g ⇑h
          y : LocalizedModule S M₁
          m : M₁
          s : Subtype fun x => Membership.mem S x
          hy : Exists fun u => Eq (HSMul.hSMul u (h m)) (HSMul.hSMul u 0)
          a : R
          aS : Membership.mem S a
          ha : Eq (HSMul.hSMul ⟨a, aS⟩ (h m)) (HSMul.hSMul ⟨a, aS⟩ 0)
          ⊢ Membership.mem (Set.range ⇑((IsLocalizedModule.map S (LocalizedModule.mkLine …
        -/
        rw [smul_zero, mk_smul, ← LinearMap.map_smul, ex (a • m)] at ha
        /-
          case intro.intro
          R : Type u_1
          inst✝⁶ : CommSemiring R
          S : Submonoid R
          M₀ : Type u_2
          inst✝⁵ : AddCommMonoid M₀
          inst✝⁴ : Module R M₀
          M₁ : Type u_3
          inst✝³ : AddCommMonoid M₁
          inst✝² : Module R M₁
          M₂ : Type u_4
          inst✝¹ : AddCommMonoid M₂
          inst✝ : Module R M₂
          g : LinearMap (RingHom.id R) M₀ M₁
          h : LinearMap (RingHom.id R) M₁ M₂
          ex : Function.Exact ⇑g ⇑h
          y : LocalizedModule S M₁
          m : M₁
          s : Subtype fun x => Membership.mem S x
          hy : Exists fun u => Eq (HSMul.hSMul u (h m)) (HSMul.hSMul u 0)
          a : R
          aS : Membership.mem S a
          ha : Membership.mem (Set.range ⇑g) (HSMul.hSMul a m)
          ⊢ Membership.mem (Set.range ⇑((IsLocalizedModule.map S (LocalizedModule.mkLine …
        -/
        rcases ha with ⟨x, hx⟩
        /-
          case intro.intro.intro
          R : Type u_1
          inst✝⁶ : CommSemiring R
          S : Submonoid R
          M₀ : Type u_2
          inst✝⁵ : AddCommMonoid M₀
          inst✝⁴ : Module R M₀
          M₁ : Type u_3
          inst✝³ : AddCommMonoid M₁
          inst✝² : Module R M₁
          M₂ : Type u_4
          inst✝¹ : AddCommMonoid M₂
          inst✝ : Module R M₂
          g : LinearMap (RingHom.id R) M₀ M₁
          h : LinearMap (RingHom.id R) M₁ M₂
          ex : Function.Exact ⇑g ⇑h
          y : LocalizedModule S M₁
          m : M₁
          s : Subtype fun x => Membership.mem S x
          hy : Exists fun u => Eq (HSMul.hSMul u (h m)) (HSMul.hSMul u 0)
          a : R
          aS : Membership.mem S a
          x : M₀
          hx : Eq (g x) (HSMul.hSMul a m)
          ⊢ Membership.mem (Set.range ⇑((IsLocalizedModule.map S (LocalizedModule.mkLine …
        -/
        use mk x (⟨a, aS⟩ * s)
        /-
          case h
          R : Type u_1
          inst✝⁶ : CommSemiring R
          S : Submonoid R
          M₀ : Type u_2
          inst✝⁵ : AddCommMonoid M₀
          inst✝⁴ : Module R M₀
          M₁ : Type u_3
          inst✝³ : AddCommMonoid M₁
          inst✝² : Module R M₁
          M₂ : Type u_4
          inst✝¹ : AddCommMonoid M₂
          inst✝ : Module R M₂
          g : LinearMap (RingHom.id R) M₀ M₁
          h : LinearMap (RingHom.id R) M₁ M₂
          ex : Function.Exact ⇑g ⇑h
          y : LocalizedModule S M₁
          m : M₁
          s : Subtype fun x => Membership.mem S x
          hy : Exists fun u => Eq (HSMul.hSMul u (h m)) (HSMul.hSMul u 0)
          a : R
          aS : Membership.mem S a
          x : M₀
          hx : Eq (g x) (HSMul.hSMul a m)
          ⊢ Eq (((IsLocalizedModule.map S (LocalizedModule.mkLinearMap S M₀) (LocalizedM …
        -/
        rw [map_LocalizedModules, hx, ← mk_cancel_common_left ⟨a, aS⟩ s m, mk_smul])
        /-
          🎉 no goals
        -/
      y)
    fun ⟨x, hx⟩ ↦ by
      /-
        R : Type u_1
        inst✝⁶ : CommSemiring R
        S : Submonoid R
        M₀ : Type u_2
        inst✝⁵ : AddCommMonoid M₀
        inst✝⁴ : Module R M₀
        M₁ : Type u_3
        inst✝³ : AddCommMonoid M₁
        inst✝² : Module R M₁
        M₂ : Type u_4
        inst✝¹ : AddCommMonoid M₂
        inst✝ : Module R M₂
        g : LinearMap (RingHom.id R) M₀ M₁
        h : LinearMap (RingHom.id R) M₁ M₂
        ex : Function.Exact ⇑g ⇑h
        y : LocalizedModule S M₁
        x✝ : Membership.mem (Set.range ⇑((IsLocalizedModule.map S (LocalizedModule.mkL …
        x : LocalizedModule S M₀
        hx : Eq (((IsLocalizedModule.map S (LocalizedModule.mkLinearMap S M₀) (Localiz …
        ⊢ Eq (((IsLocalizedModule.map S (LocalizedModule.mkLinearMap S M₁) (LocalizedM …
      -/
      revert hx
      /-
        R : Type u_1
        inst✝⁶ : CommSemiring R
        S : Submonoid R
        M₀ : Type u_2
        inst✝⁵ : AddCommMonoid M₀
        inst✝⁴ : Module R M₀
        M₁ : Type u_3
        inst✝³ : AddCommMonoid M₁
        inst✝² : Module R M₁
        M₂ : Type u_4
        inst✝¹ : AddCommMonoid M₂
        inst✝ : Module R M₂
        g : LinearMap (RingHom.id R) M₀ M₁
        h : LinearMap (RingHom.id R) M₁ M₂
        ex : Function.Exact ⇑g ⇑h
        y : LocalizedModule S M₁
        x✝ : Membership.mem (Set.range ⇑((IsLocalizedModule.map S (LocalizedModule.mkL …
        x : LocalizedModule S M₀
        ⊢ Eq (((IsLocalizedModule.map S (LocalizedModule.mkLinearMap S M₀) (LocalizedM …
      -/
      refine induction_on (fun m s hx ↦ ?_) x
      /-
        R : Type u_1
        inst✝⁶ : CommSemiring R
        S : Submonoid R
        M₀ : Type u_2
        inst✝⁵ : AddCommMonoid M₀
        inst✝⁴ : Module R M₀
        M₁ : Type u_3
        inst✝³ : AddCommMonoid M₁
        inst✝² : Module R M₁
        M₂ : Type u_4
        inst✝¹ : AddCommMonoid M₂
        inst✝ : Module R M₂
        g : LinearMap (RingHom.id R) M₀ M₁
        h : LinearMap (RingHom.id R) M₁ M₂
        ex : Function.Exact ⇑g ⇑h
        y : LocalizedModule S M₁
        x✝ : Membership.mem (Set.range ⇑((IsLocalizedModule.map S (LocalizedModule.mkL …
        x : LocalizedModule S M₀
        m : M₀
        s : Subtype fun x => Membership.mem S x
        hx : Eq (((IsLocalizedModule.map S (LocalizedModule.mkLinearMap S M₀) (Localiz …
        ⊢ Eq (((IsLocalizedModule.map S (LocalizedModule.mkLinearMap S M₁) (LocalizedM …
      -/
      rw [← hx, map_LocalizedModules, map_LocalizedModules, (ex (g m)).2 ⟨m, rfl⟩, zero_mk]
      /-
        🎉 no goals
      -/


/-- Localization of modules is an exact functor. -/
theorem IsLocalizedModule.map_exact (g : M₀ →ₗ[R] M₁) (h : M₁ →ₗ[R] M₂) (ex : Function.Exact g h) :
    Function.Exact (map S f₀ f₁ g) (map S f₁ f₂ h) :=
  Function.Exact.of_ladder_linearEquiv_of_exact
    (map_iso_commute S f₀ f₁ g) (map_iso_commute S f₁ f₂ h) (LocalizedModule.map_exact S g h ex)


