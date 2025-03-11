/-- Given a linear map `f : M₁ →ₗ[A] M₂`, a presentation of `M₂` and a choice
of generators of `M₁`, this structure specifies a lifting of the image by `f`
of each generator of `M₁` as a linear combination of the generators of `M₂`. -/
structure CokernelData where
  /-- a lifting of `f (g₁ i)` in `pres₂.G →₀ A` -/
  lift (i : ι) : pres₂.G →₀ A
  π_lift (i : ι) : pres₂.π (lift i) = f (g₁ i)


/-- Constructor for `Presentation.CokernelData` in case we have a chosen set-theoretic
section of the projection `(pres₂.G →₀ A) → M₂`. -/
@[simps]
def CokernelData.ofSection (s : M₂ → (pres₂.G →₀ A))
    (hs : ∀ (m₂ : M₂), pres₂.π (s m₂) = m₂) :
    pres₂.CokernelData f g₁ where
  lift i := s (f (g₁ i))
                 /-
                   A : Type u
                   inst✝⁶ : Ring A
                   M₁ : Type v₁
                   M₂ : Type v₂
                   M₃ : Type v₃
                   inst✝⁵ : AddCommGroup M₁
                   inst✝⁴ : Module A M₁
                   inst✝³ : AddCommGroup M₂
                   inst✝² : Module A M₂
                   inst✝¹ : AddCommGroup M₃
                   inst✝ : Module A M₃
                   pres₂ : Module.Presentation A M₂
                   f : LinearMap (RingHom.id A) M₁ M₂
                   ι : Type w₁
                   g₁ : ι → M₁
                   s : M₂ → Finsupp pres₂.G A
                   hs : ∀ (m₂ : M₂), Eq (pres₂.π (s m₂)) m₂
                   i : ι
                   ⊢ Eq (pres₂.π ((fun i => s (f (g₁ i))) i)) (f (g₁ i))
                 -/
  π_lift i := by simp [hs]
                 /-
                   🎉 no goals
                 -/


instance nonempty_cokernelData :
    Nonempty (pres₂.CokernelData f g₁) := by
  /-
    A : Type u
    inst✝⁶ : Ring A
    M₁ : Type v₁
    M₂ : Type v₂
    M₃ : Type v₃
    inst✝⁵ : AddCommGroup M₁
    inst✝⁴ : Module A M₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module A M₂
    inst✝¹ : AddCommGroup M₃
    inst✝ : Module A M₃
    pres₂ : Module.Presentation A M₂
    f : LinearMap (RingHom.id A) M₁ M₂
    ι : Type w₁
    g₁ : ι → M₁
    ⊢ Nonempty (pres₂.CokernelData f g₁)
  -/
  obtain ⟨s, hs⟩ := pres₂.surjective_π.hasRightInverse
  /-
    case intro
    A : Type u
    inst✝⁶ : Ring A
    M₁ : Type v₁
    M₂ : Type v₂
    M₃ : Type v₃
    inst✝⁵ : AddCommGroup M₁
    inst✝⁴ : Module A M₁
    inst✝³ : AddCommGroup M₂
    inst✝² : Module A M₂
    inst✝¹ : AddCommGroup M₃
    inst✝ : Module A M₃
    pres₂ : Module.Presentation A M₂
    f : LinearMap (RingHom.id A) M₁ M₂
    ι : Type w₁
    g₁ : ι → M₁
    s : M₂ → Finsupp pres₂.G A
    hs : Function.RightInverse s ⇑pres₂.π
    ⊢ Nonempty (pres₂.CokernelData f g₁)
  -/
  exact ⟨CokernelData.ofSection _ _ _ s hs⟩
  /-
    🎉 no goals
  -/


/-- The shape of the presentation by generators and relations of the cokernel
of `f : M₁ →ₗ[A] M₂`. It consists of a generator for each generator of `M₂`, and
there are two types of relations: one for each relation in the presentation in `M₂`,
and one for each generator of `M₁`. -/
@[simps]
def cokernelRelations : Relations A where
  G := pres₂.G
  R := Sum pres₂.R ι
  relation x := match x with
    | .inl r => pres₂.relation r
    | .inr i => data.lift i


/-- The obvious solution in `M₂ ⧸ LinearMap.range f` to the equations in
`pres₂.cokernelRelations data`. -/
@[simps]
def cokernelSolution :
    (pres₂.cokernelRelations data).Solution (M₂ ⧸ LinearMap.range f) where
  var g := Submodule.mkQ _ (pres₂.var g)
  linearCombination_var_relation := by
    /-
      A : Type u
      inst✝⁶ : Ring A
      M₁ : Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      inst✝⁵ : AddCommGroup M₁
      inst✝⁴ : Module A M₁
      inst✝³ : AddCommGroup M₂
      inst✝² : Module A M₂
      inst✝¹ : AddCommGroup M₃
      inst✝ : Module A M₃
      pres₂ : Module.Presentation A M₂
      f : LinearMap (RingHom.id A) M₁ M₂
      ι : Type w₁
      g₁ : ι → M₁
      data : pres₂.CokernelData f g₁
      ⊢ ∀ (r : (pres₂.cokernelRelations data).R), Eq ((Finsupp.linearCombination A f …
    -/
    intro x
    /-
      A : Type u
      inst✝⁶ : Ring A
      M₁ : Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      inst✝⁵ : AddCommGroup M₁
      inst✝⁴ : Module A M₁
      inst✝³ : AddCommGroup M₂
      inst✝² : Module A M₂
      inst✝¹ : AddCommGroup M₃
      inst✝ : Module A M₃
      pres₂ : Module.Presentation A M₂
      f : LinearMap (RingHom.id A) M₁ M₂
      ι : Type w₁
      g₁ : ι → M₁
      data : pres₂.CokernelData f g₁
      x : (pres₂.cokernelRelations data).R
      ⊢ Eq ((Finsupp.linearCombination A fun g => (LinearMap.range f).mkQ (pres₂.var …
    -/
    erw [← Finsupp.apply_linearCombination]
    /-
      A : Type u
      inst✝⁶ : Ring A
      M₁ : Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      inst✝⁵ : AddCommGroup M₁
      inst✝⁴ : Module A M₁
      inst✝³ : AddCommGroup M₂
      inst✝² : Module A M₂
      inst✝¹ : AddCommGroup M₃
      inst✝ : Module A M₃
      pres₂ : Module.Presentation A M₂
      f : LinearMap (RingHom.id A) M₁ M₂
      ι : Type w₁
      g₁ : ι → M₁
      data : pres₂.CokernelData f g₁
      x : (pres₂.cokernelRelations data).R
      ⊢ Eq ((LinearMap.range f).mkQ ((Finsupp.linearCombination A pres₂.var) ((pres₂ …
    -/
    obtain (r | i) := x
      /-
        case inl
        A : Type u
        inst✝⁶ : Ring A
        M₁ : Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        inst✝⁵ : AddCommGroup M₁
        inst✝⁴ : Module A M₁
        inst✝³ : AddCommGroup M₂
        inst✝² : Module A M₂
        inst✝¹ : AddCommGroup M₃
        inst✝ : Module A M₃
        pres₂ : Module.Presentation A M₂
        f : LinearMap (RingHom.id A) M₁ M₂
        ι : Type w₁
        g₁ : ι → M₁
        data : pres₂.CokernelData f g₁
        r : pres₂.R
        ⊢ Eq ((LinearMap.range f).mkQ ((Finsupp.linearCombination A pres₂.var) ((pres₂ …
      -/
    · erw [pres₂.linearCombination_var_relation]
      /-
        case inl
        A : Type u
        inst✝⁶ : Ring A
        M₁ : Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        inst✝⁵ : AddCommGroup M₁
        inst✝⁴ : Module A M₁
        inst✝³ : AddCommGroup M₂
        inst✝² : Module A M₂
        inst✝¹ : AddCommGroup M₃
        inst✝ : Module A M₃
        pres₂ : Module.Presentation A M₂
        f : LinearMap (RingHom.id A) M₁ M₂
        ι : Type w₁
        g₁ : ι → M₁
        data : pres₂.CokernelData f g₁
        r : pres₂.R
        ⊢ Eq ((LinearMap.range f).mkQ 0) 0
      -/
      dsimp
      /-
        🎉 no goals
      -/
      /-
        case inr
        A : Type u
        inst✝⁶ : Ring A
        M₁ : Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        inst✝⁵ : AddCommGroup M₁
        inst✝⁴ : Module A M₁
        inst✝³ : AddCommGroup M₂
        inst✝² : Module A M₂
        inst✝¹ : AddCommGroup M₃
        inst✝ : Module A M₃
        pres₂ : Module.Presentation A M₂
        f : LinearMap (RingHom.id A) M₁ M₂
        ι : Type w₁
        g₁ : ι → M₁
        data : pres₂.CokernelData f g₁
        i : ι
        ⊢ Eq ((LinearMap.range f).mkQ ((Finsupp.linearCombination A pres₂.var) ((pres₂ …
      -/
    · erw [data.π_lift]
      /-
        case inr
        A : Type u
        inst✝⁶ : Ring A
        M₁ : Type v₁
        M₂ : Type v₂
        M₃ : Type v₃
        inst✝⁵ : AddCommGroup M₁
        inst✝⁴ : Module A M₁
        inst✝³ : AddCommGroup M₂
        inst✝² : Module A M₂
        inst✝¹ : AddCommGroup M₃
        inst✝ : Module A M₃
        pres₂ : Module.Presentation A M₂
        f : LinearMap (RingHom.id A) M₁ M₂
        ι : Type w₁
        g₁ : ι → M₁
        data : pres₂.CokernelData f g₁
        i : ι
        ⊢ Eq ((LinearMap.range f).mkQ (f (g₁ i))) 0
      -/
      simp
      /-
        🎉 no goals
      -/


/-- The cokernel can be defined by generators and relations. -/
noncomputable def isPresentationCore :
    Relations.Solution.IsPresentationCore.{w}
      (pres₂.cokernelSolution data) where
  desc s := (LinearMap.range f).liftQ (pres₂.desc
    { var := s.var
      linearCombination_var_relation :=
        fun r ↦ s.linearCombination_var_relation (.inl r) }) (by
          rw [LinearMap.range_eq_map, ← hg₁, Submodule.map_span, Submodule.span_le,
            Set.image_subset_iff]
          /-
            A : Type u
            inst✝⁸ : Ring A
            M₁ : Type v₁
            M₂ : Type v₂
            M₃ : Type v₃
            inst✝⁷ : AddCommGroup M₁
            inst✝⁶ : Module A M₁
            inst✝⁵ : AddCommGroup M₂
            inst✝⁴ : Module A M₂
            inst✝³ : AddCommGroup M₃
            inst✝² : Module A M₃
            pres₂ : Module.Presentation A M₂
            f : LinearMap (RingHom.id A) M₁ M₂
            ι : Type w₁
            g₁ : ι → M₁
            data : pres₂.CokernelData f g₁
            hg₁ : Eq (Submodule.span A (Set.range g₁)) Top.top
            N✝ : Type w
            inst✝¹ : AddCommGroup N✝
            inst✝ : Module A N✝
            s : (pres₂.cokernelRelations data).Solution N✝
            ⊢ HasSubset.Subset (Set.range g₁) (Set.preimage ⇑f ↑(LinearMap.ker (⋯.desc { v …
          -/
          rintro _ ⟨i, rfl⟩
          rw [Set.mem_preimage, SetLike.mem_coe, LinearMap.mem_ker, ← data.π_lift,
            Relations.Solution.IsPresentation.π_desc_apply]
          /-
            case intro
            A : Type u
            inst✝⁸ : Ring A
            M₁ : Type v₁
            M₂ : Type v₂
            M₃ : Type v₃
            inst✝⁷ : AddCommGroup M₁
            inst✝⁶ : Module A M₁
            inst✝⁵ : AddCommGroup M₂
            inst✝⁴ : Module A M₂
            inst✝³ : AddCommGroup M₃
            inst✝² : Module A M₃
            pres₂ : Module.Presentation A M₂
            f : LinearMap (RingHom.id A) M₁ M₂
            ι : Type w₁
            g₁ : ι → M₁
            data : pres₂.CokernelData f g₁
            hg₁ : Eq (Submodule.span A (Set.range g₁)) Top.top
            N✝ : Type w
            inst✝¹ : AddCommGroup N✝
            inst✝ : Module A N✝
            s : (pres₂.cokernelRelations data).Solution N✝
            i : ι
            ⊢ Eq ({ var := s.var, linearCombination_var_relation := ⋯ }.π (data.lift i)) 0
          -/
          exact s.linearCombination_var_relation (.inr i))
          /-
            🎉 no goals
          -/
                        /-
                          A : Type u
                          inst✝⁸ : Ring A
                          M₁ : Type v₁
                          M₂ : Type v₂
                          M₃ : Type v₃
                          inst✝⁷ : AddCommGroup M₁
                          inst✝⁶ : Module A M₁
                          inst✝⁵ : AddCommGroup M₂
                          inst✝⁴ : Module A M₂
                          inst✝³ : AddCommGroup M₃
                          inst✝² : Module A M₃
                          pres₂ : Module.Presentation A M₂
                          f : LinearMap (RingHom.id A) M₁ M₂
                          ι : Type w₁
                          g₁ : ι → M₁
                          data : pres₂.CokernelData f g₁
                          hg₁ : Eq (Submodule.span A (Set.range g₁)) Top.top
                          N✝ : Type w
                          inst✝¹ : AddCommGroup N✝
                          inst✝ : Module A N✝
                          s : (pres₂.cokernelRelations data).Solution N✝
                          ⊢ Eq ((pres₂.cokernelSolution data).postcomp ((fun {N} [AddCommGroup N] [Modul …
                        -/
  postcomp_desc s := by aesop
                        /-
                          🎉 no goals
                        -/
  postcomp_injective h := by
    /-
      A : Type u
      inst✝⁸ : Ring A
      M₁ : Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      inst✝⁷ : AddCommGroup M₁
      inst✝⁶ : Module A M₁
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module A M₂
      inst✝³ : AddCommGroup M₃
      inst✝² : Module A M₃
      pres₂ : Module.Presentation A M₂
      f : LinearMap (RingHom.id A) M₁ M₂
      ι : Type w₁
      g₁ : ι → M₁
      data : pres₂.CokernelData f g₁
      hg₁ : Eq (Submodule.span A (Set.range g₁)) Top.top
      N✝ : Type w
      inst✝¹ : AddCommGroup N✝
      inst✝ : Module A N✝
      f✝ f'✝ : LinearMap (RingHom.id A) (HasQuotient.Quotient M₂ (LinearMap.range f) …
      h : Eq ((pres₂.cokernelSolution data).postcomp f✝) ((pres₂.cokernelSolution da …
      ⊢ Eq f✝ f'✝
    -/
    ext : 1
    /-
      case h
      A : Type u
      inst✝⁸ : Ring A
      M₁ : Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      inst✝⁷ : AddCommGroup M₁
      inst✝⁶ : Module A M₁
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module A M₂
      inst✝³ : AddCommGroup M₃
      inst✝² : Module A M₃
      pres₂ : Module.Presentation A M₂
      f : LinearMap (RingHom.id A) M₁ M₂
      ι : Type w₁
      g₁ : ι → M₁
      data : pres₂.CokernelData f g₁
      hg₁ : Eq (Submodule.span A (Set.range g₁)) Top.top
      N✝ : Type w
      inst✝¹ : AddCommGroup N✝
      inst✝ : Module A N✝
      f✝ f'✝ : LinearMap (RingHom.id A) (HasQuotient.Quotient M₂ (LinearMap.range f) …
      h : Eq ((pres₂.cokernelSolution data).postcomp f✝) ((pres₂.cokernelSolution da …
      ⊢ Eq (f✝.comp (LinearMap.range f).mkQ) (f'✝.comp (LinearMap.range f).mkQ)
    -/
    apply pres₂.toIsPresentation.postcomp_injective
    /-
      case h
      A : Type u
      inst✝⁸ : Ring A
      M₁ : Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      inst✝⁷ : AddCommGroup M₁
      inst✝⁶ : Module A M₁
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module A M₂
      inst✝³ : AddCommGroup M₃
      inst✝² : Module A M₃
      pres₂ : Module.Presentation A M₂
      f : LinearMap (RingHom.id A) M₁ M₂
      ι : Type w₁
      g₁ : ι → M₁
      data : pres₂.CokernelData f g₁
      hg₁ : Eq (Submodule.span A (Set.range g₁)) Top.top
      N✝ : Type w
      inst✝¹ : AddCommGroup N✝
      inst✝ : Module A N✝
      f✝ f'✝ : LinearMap (RingHom.id A) (HasQuotient.Quotient M₂ (LinearMap.range f) …
      h : Eq ((pres₂.cokernelSolution data).postcomp f✝) ((pres₂.cokernelSolution da …
      ⊢ Eq (pres₂.postcomp (f✝.comp (LinearMap.range f).mkQ)) (pres₂.postcomp (f'✝.c …
    -/
    ext g
    /-
      case h.var.h
      A : Type u
      inst✝⁸ : Ring A
      M₁ : Type v₁
      M₂ : Type v₂
      M₃ : Type v₃
      inst✝⁷ : AddCommGroup M₁
      inst✝⁶ : Module A M₁
      inst✝⁵ : AddCommGroup M₂
      inst✝⁴ : Module A M₂
      inst✝³ : AddCommGroup M₃
      inst✝² : Module A M₃
      pres₂ : Module.Presentation A M₂
      f : LinearMap (RingHom.id A) M₁ M₂
      ι : Type w₁
      g₁ : ι → M₁
      data : pres₂.CokernelData f g₁
      hg₁ : Eq (Submodule.span A (Set.range g₁)) Top.top
      N✝ : Type w
      inst✝¹ : AddCommGroup N✝
      inst✝ : Module A N✝
      f✝ f'✝ : LinearMap (RingHom.id A) (HasQuotient.Quotient M₂ (LinearMap.range f) …
      h : Eq ((pres₂.cokernelSolution data).postcomp f✝) ((pres₂.cokernelSolution da …
      g : pres₂.G
      ⊢ Eq ((pres₂.postcomp (f✝.comp (LinearMap.range f).mkQ)).var g) ((pres₂.postco …
    -/
    exact Relations.Solution.congr_var h g
    /-
      🎉 no goals
    -/


include hg₁ in
lemma isPresentation : (pres₂.cokernelSolution data).IsPresentation :=
  (isPresentationCore pres₂ data hg₁).isPresentation


/-- The presentation of the cokernel of a linear map `f : M₁ →ₗ[A] M₂` that is obtained
from a presentation `pres₂` of `M₂`, a choice of generators `g₁ : ι → M₁` of `M₁`,
and an additional data in `pres₂.CokernelData f g₁`. -/
@[simps!]
def cokernel : Presentation A (M₂ ⧸ LinearMap.range f) :=
  ofIsPresentation (cokernelSolution.isPresentation pres₂ data hg₁)


/-- Given an exact sequence of `A`-modules `M₁ → M₂ → M₃ → 0`, this is the presentation
of `M₃` that is obtained from a presentation `pres₂` of `M₂`, a choice of generators
`g₁ : ι → M₁` of `M₁`, and an additional data in a `Presentation.CokernelData` structure. -/
@[simps!]
noncomputable def ofExact {f : M₁ →ₗ[A] M₂} {g : M₂ →ₗ[A] M₃}
    (pres₂ : Presentation.{w₂₀, w₂₁} A M₂) {ι : Type w₁} {g₁ : ι → M₁}
    (data : pres₂.CokernelData f g₁)
    (hfg : Function.Exact f g) (hg : Function.Surjective g)
    (hg₁ : Submodule.span A (Set.range g₁) = ⊤) :
    Presentation A M₃ :=
  (pres₂.cokernel data hg₁).ofLinearEquiv (hfg.linearEquivOfSurjective hg)


