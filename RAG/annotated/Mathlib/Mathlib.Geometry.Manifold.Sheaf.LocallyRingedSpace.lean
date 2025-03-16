local notation "∞" => (⊤ : ℕ∞)


/-- The units of the stalk at `x` of the sheaf of smooth functions from `M` to `𝕜`, considered as a
sheaf of commutative rings, are the functions whose values at `x` are nonzero. -/
theorem smoothSheafCommRing.isUnit_stalk_iff {x : M}
    (f : (smoothSheafCommRing IM 𝓘(𝕜) M 𝕜).presheaf.stalk x) :
    IsUnit f ↔ f ∉ RingHom.ker (smoothSheafCommRing.eval IM 𝓘(𝕜) M 𝕜 x) := by
  /-
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    EM : Type u_1
    inst✝⁴ : NormedAddCommGroup EM
    inst✝³ : NormedSpace 𝕜 EM
    HM : Type u_2
    inst✝² : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    M : Type u
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace HM M
    x : M
    f : ↑((smoothSheafCommRing IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜).presheaf.stalk x)
    ⊢ Iff (IsUnit f) (Not (Membership.mem (RingHom.ker (smoothSheafCommRing.eval I …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      EM : Type u_1
      inst✝⁴ : NormedAddCommGroup EM
      inst✝³ : NormedSpace 𝕜 EM
      HM : Type u_2
      inst✝² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      M : Type u
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace HM M
      x : M
      f : ↑((smoothSheafCommRing IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜).presheaf.stalk x)
      ⊢ IsUnit f → Not (Membership.mem (RingHom.ker (smoothSheafCommRing.eval IM (mo …
    -/
  · rintro ⟨⟨f, g, hf, hg⟩, rfl⟩ (h' : smoothSheafCommRing.eval IM 𝓘(𝕜) M 𝕜 x f = 0)
    /-
      case mp.intro.mk
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      EM : Type u_1
      inst✝⁴ : NormedAddCommGroup EM
      inst✝³ : NormedSpace 𝕜 EM
      HM : Type u_2
      inst✝² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      M : Type u
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace HM M
      x : M
      f g : ↑((smoothSheafCommRing IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜).presheaf.stalk …
      hf : Eq (HMul.hMul f g) 1
      hg : Eq (HMul.hMul g f) 1
      h' : Eq ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) f) 0
      ⊢ False
    -/
    simpa [h'] using congr_arg (smoothSheafCommRing.eval IM 𝓘(𝕜) M 𝕜 x) hf
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      EM : Type u_1
      inst✝⁴ : NormedAddCommGroup EM
      inst✝³ : NormedSpace 𝕜 EM
      HM : Type u_2
      inst✝² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      M : Type u
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace HM M
      x : M
      f : ↑((smoothSheafCommRing IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜).presheaf.stalk x)
      ⊢ Not (Membership.mem (RingHom.ker (smoothSheafCommRing.eval IM (modelWithCorn …
    -/
  · let S := (smoothSheafCommRing IM 𝓘(𝕜) M 𝕜).presheaf
    -- Suppose that `f`, in the stalk at `x`, is nonzero at `x`
    /-
      case mpr
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      EM : Type u_1
      inst✝⁴ : NormedAddCommGroup EM
      inst✝³ : NormedSpace 𝕜 EM
      HM : Type u_2
      inst✝² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      M : Type u
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace HM M
      x : M
      f : ↑((smoothSheafCommRing IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜).presheaf.stalk x)
      S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
      ⊢ Not (Membership.mem (RingHom.ker (smoothSheafCommRing.eval IM (modelWithCorn …
    -/
    rintro (hf : _ ≠ 0)
    -- Represent `f` as the germ of some function (also called `f`) on an open neighbourhood `U` of
    -- `x`, which is nonzero at `x`
    /-
      case mpr
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      EM : Type u_1
      inst✝⁴ : NormedAddCommGroup EM
      inst✝³ : NormedSpace 𝕜 EM
      HM : Type u_2
      inst✝² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      M : Type u
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace HM M
      x : M
      f : ↑((smoothSheafCommRing IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜).presheaf.stalk x)
      S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
      hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) f) 0
      ⊢ IsUnit f
    -/
    obtain ⟨U : Opens M, hxU, f : C^∞⟮IM, U; 𝓘(𝕜), 𝕜⟯, rfl⟩ := S.germ_exist x f
    have hf' : f ⟨x, hxU⟩ ≠ 0 := by
      convert hf
      exact (smoothSheafCommRing.eval_germ U x hxU f).symm
    -- In fact, by continuity, `f` is nonzero on a neighbourhood `V` of `x`
    /-
      case mpr.intro.intro.intro
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      EM : Type u_1
      inst✝⁴ : NormedAddCommGroup EM
      inst✝³ : NormedSpace 𝕜 EM
      HM : Type u_2
      inst✝² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      M : Type u
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace HM M
      x : M
      S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
      U : TopologicalSpace.Opens M
      hxU : Membership.mem U x
      f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
      hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
      hf' : Ne (f ⟨x, hxU⟩) 0
      ⊢ IsUnit ((S.germ U x hxU) f)
    -/
    have H :  ∀ᶠ (z : U) in 𝓝 ⟨x, hxU⟩, f z ≠ 0 := f.2.continuous.continuousAt.eventually_ne hf'
    /-
      case mpr.intro.intro.intro
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      EM : Type u_1
      inst✝⁴ : NormedAddCommGroup EM
      inst✝³ : NormedSpace 𝕜 EM
      HM : Type u_2
      inst✝² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      M : Type u
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace HM M
      x : M
      S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
      U : TopologicalSpace.Opens M
      hxU : Membership.mem U x
      f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
      hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
      hf' : Ne (f ⟨x, hxU⟩) 0
      H : Filter.Eventually (fun z => Ne (f z) 0) (nhds ⟨x, hxU⟩)
      ⊢ IsUnit ((S.germ U x hxU) f)
    -/
    rw [eventually_nhds_iff] at H
    /-
      case mpr.intro.intro.intro
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      EM : Type u_1
      inst✝⁴ : NormedAddCommGroup EM
      inst✝³ : NormedSpace 𝕜 EM
      HM : Type u_2
      inst✝² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      M : Type u
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace HM M
      x : M
      S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
      U : TopologicalSpace.Opens M
      hxU : Membership.mem U x
      f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
      hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
      hf' : Ne (f ⟨x, hxU⟩) 0
      H : Exists fun t => And (∀ (y : Subtype fun x => Membership.mem U x), Membersh …
      ⊢ IsUnit ((S.germ U x hxU) f)
    -/
    obtain ⟨V₀, hV₀f, hV₀, hxV₀⟩ := H
    /-
      case mpr.intro.intro.intro.intro.intro.intro
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      EM : Type u_1
      inst✝⁴ : NormedAddCommGroup EM
      inst✝³ : NormedSpace 𝕜 EM
      HM : Type u_2
      inst✝² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      M : Type u
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace HM M
      x : M
      S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
      U : TopologicalSpace.Opens M
      hxU : Membership.mem U x
      f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
      hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
      hf' : Ne (f ⟨x, hxU⟩) 0
      V₀ : Set (Subtype fun x => Membership.mem U x)
      hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem V₀ y → Ne ( …
      hV₀ : IsOpen V₀
      hxV₀ : Membership.mem V₀ ⟨x, hxU⟩
      ⊢ IsUnit ((S.germ U x hxU) f)
    -/
    let V : Opens M := ⟨Subtype.val '' V₀, U.2.isOpenMap_subtype_val V₀ hV₀⟩
    /-
      case mpr.intro.intro.intro.intro.intro.intro
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      EM : Type u_1
      inst✝⁴ : NormedAddCommGroup EM
      inst✝³ : NormedSpace 𝕜 EM
      HM : Type u_2
      inst✝² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      M : Type u
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace HM M
      x : M
      S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
      U : TopologicalSpace.Opens M
      hxU : Membership.mem U x
      f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
      hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
      hf' : Ne (f ⟨x, hxU⟩) 0
      V₀ : Set (Subtype fun x => Membership.mem U x)
      hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem V₀ y → Ne ( …
      hV₀ : IsOpen V₀
      hxV₀ : Membership.mem V₀ ⟨x, hxU⟩
      V : TopologicalSpace.Opens M := { carrier := Set.image Subtype.val V₀, is_open …
      ⊢ IsUnit ((S.germ U x hxU) f)
    -/
    have hUV : V ≤ U := Subtype.coe_image_subset (U : Set M) V₀
    have hV : V₀ = Set.range (Set.inclusion hUV) := by
      convert (Set.range_inclusion hUV).symm
      ext y
      show _ ↔ y ∈ Subtype.val ⁻¹' (Subtype.val '' V₀)
      rw [Set.preimage_image_eq _ Subtype.coe_injective]
    /-
      case mpr.intro.intro.intro.intro.intro.intro
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      EM : Type u_1
      inst✝⁴ : NormedAddCommGroup EM
      inst✝³ : NormedSpace 𝕜 EM
      HM : Type u_2
      inst✝² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      M : Type u
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace HM M
      x : M
      S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
      U : TopologicalSpace.Opens M
      hxU : Membership.mem U x
      f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
      hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
      hf' : Ne (f ⟨x, hxU⟩) 0
      V₀ : Set (Subtype fun x => Membership.mem U x)
      hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem V₀ y → Ne ( …
      hV₀ : IsOpen V₀
      hxV₀ : Membership.mem V₀ ⟨x, hxU⟩
      V : TopologicalSpace.Opens M := { carrier := Set.image Subtype.val V₀, is_open …
      hUV : LE.le V U
      hV : Eq V₀ (Set.range (Set.inclusion hUV))
      ⊢ IsUnit ((S.germ U x hxU) f)
    -/
    clear_value V
    /-
      case mpr.intro.intro.intro.intro.intro.intro
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      EM : Type u_1
      inst✝⁴ : NormedAddCommGroup EM
      inst✝³ : NormedSpace 𝕜 EM
      HM : Type u_2
      inst✝² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      M : Type u
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace HM M
      x : M
      S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
      U : TopologicalSpace.Opens M
      hxU : Membership.mem U x
      f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
      hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
      hf' : Ne (f ⟨x, hxU⟩) 0
      V₀ : Set (Subtype fun x => Membership.mem U x)
      hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem V₀ y → Ne ( …
      hV₀ : IsOpen V₀
      hxV₀ : Membership.mem V₀ ⟨x, hxU⟩
      V : TopologicalSpace.Opens M
      hUV : LE.le V U
      hV : Eq V₀ (Set.range (Set.inclusion hUV))
      ⊢ IsUnit ((S.germ U x hxU) f)
    -/
    subst hV
    have hxV : x ∈ (V : Set M) := by
      obtain ⟨x₀, hxx₀⟩ := hxV₀
      convert x₀.2
      exact congr_arg Subtype.val hxx₀.symm
    have hVf : ∀ y : V, f (Set.inclusion hUV y) ≠ 0 :=
      fun y ↦ hV₀f (Set.inclusion hUV y) (Set.mem_range_self y)
    -- Let `g` be the pointwise inverse of `f` on `V`, which is smooth since `f` is nonzero there
    /-
      case mpr.intro.intro.intro.intro.intro.intro
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      EM : Type u_1
      inst✝⁴ : NormedAddCommGroup EM
      inst✝³ : NormedSpace 𝕜 EM
      HM : Type u_2
      inst✝² : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      M : Type u
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace HM M
      x : M
      S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
      U : TopologicalSpace.Opens M
      hxU : Membership.mem U x
      f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
      hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
      hf' : Ne (f ⟨x, hxU⟩) 0
      V : TopologicalSpace.Opens M
      hUV : LE.le V U
      hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem (Set.range  …
      hV₀ : IsOpen (Set.range (Set.inclusion hUV))
      hxV₀ : Membership.mem (Set.range (Set.inclusion hUV)) ⟨x, hxU⟩
      hxV : Membership.mem (↑V) x
      hVf : ∀ (y : Subtype fun x => Membership.mem V x), Ne (f (Set.inclusion hUV y) …
      ⊢ IsUnit ((S.germ U x hxU) f)
    -/
    let g : C^∞⟮IM, V; 𝓘(𝕜), 𝕜⟯ := ⟨(f ∘ Set.inclusion hUV)⁻¹, ?_⟩
    -- The germ of `g` is inverse to the germ of `f`, so `f` is a unit
    · refine ⟨⟨S.germ _ x (hxV) (SmoothMap.restrictRingHom IM 𝓘(𝕜) 𝕜 hUV f), S.germ _ x hxV g,
        ?_, ?_⟩, S.germ_res_apply hUV.hom x hxV f⟩
        /-
          case mpr.intro.intro.intro.intro.intro.intro.refine_2.refine_1
          𝕜 : Type u
          inst✝⁵ : NontriviallyNormedField 𝕜
          EM : Type u_1
          inst✝⁴ : NormedAddCommGroup EM
          inst✝³ : NormedSpace 𝕜 EM
          HM : Type u_2
          inst✝² : TopologicalSpace HM
          IM : ModelWithCorners 𝕜 EM HM
          M : Type u
          inst✝¹ : TopologicalSpace M
          inst✝ : ChartedSpace HM M
          x : M
          S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
          U : TopologicalSpace.Opens M
          hxU : Membership.mem U x
          f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
          hf' : Ne (f ⟨x, hxU⟩) 0
          V : TopologicalSpace.Opens M
          hUV : LE.le V U
          hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem (Set.range  …
          hV₀ : IsOpen (Set.range (Set.inclusion hUV))
          hxV₀ : Membership.mem (Set.range (Set.inclusion hUV)) ⟨x, hxU⟩
          hxV : Membership.mem (↑V) x
          hVf : ∀ (y : Subtype fun x => Membership.mem V x), Ne (f (Set.inclusion hUV y) …
          g : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          ⊢ Eq (HMul.hMul ((S.germ V x hxV).hom ((SmoothMap.restrictRingHom IM (modelWit …
        -/
      · rw [← map_mul]
        -- Qualified the name to avoid Lean not finding a `OneHomClass` https://github.com/leanprover-community/mathlib4/pull/8386
        /-
          case mpr.intro.intro.intro.intro.intro.intro.refine_2.refine_1
          𝕜 : Type u
          inst✝⁵ : NontriviallyNormedField 𝕜
          EM : Type u_1
          inst✝⁴ : NormedAddCommGroup EM
          inst✝³ : NormedSpace 𝕜 EM
          HM : Type u_2
          inst✝² : TopologicalSpace HM
          IM : ModelWithCorners 𝕜 EM HM
          M : Type u
          inst✝¹ : TopologicalSpace M
          inst✝ : ChartedSpace HM M
          x : M
          S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
          U : TopologicalSpace.Opens M
          hxU : Membership.mem U x
          f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
          hf' : Ne (f ⟨x, hxU⟩) 0
          V : TopologicalSpace.Opens M
          hUV : LE.le V U
          hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem (Set.range  …
          hV₀ : IsOpen (Set.range (Set.inclusion hUV))
          hxV₀ : Membership.mem (Set.range (Set.inclusion hUV)) ⟨x, hxU⟩
          hxV : Membership.mem (↑V) x
          hVf : ∀ (y : Subtype fun x => Membership.mem V x), Ne (f (Set.inclusion hUV y) …
          g : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          ⊢ Eq ((S.germ V x hxV).hom (HMul.hMul ((SmoothMap.restrictRingHom IM (modelWit …
        -/
        convert RingHom.map_one _
        /-
          case h.e'_2.h.e'_6
          𝕜 : Type u
          inst✝⁵ : NontriviallyNormedField 𝕜
          EM : Type u_1
          inst✝⁴ : NormedAddCommGroup EM
          inst✝³ : NormedSpace 𝕜 EM
          HM : Type u_2
          inst✝² : TopologicalSpace HM
          IM : ModelWithCorners 𝕜 EM HM
          M : Type u
          inst✝¹ : TopologicalSpace M
          inst✝ : ChartedSpace HM M
          x : M
          S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
          U : TopologicalSpace.Opens M
          hxU : Membership.mem U x
          f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
          hf' : Ne (f ⟨x, hxU⟩) 0
          V : TopologicalSpace.Opens M
          hUV : LE.le V U
          hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem (Set.range  …
          hV₀ : IsOpen (Set.range (Set.inclusion hUV))
          hxV₀ : Membership.mem (Set.range (Set.inclusion hUV)) ⟨x, hxU⟩
          hxV : Membership.mem (↑V) x
          hVf : ∀ (y : Subtype fun x => Membership.mem V x), Ne (f (Set.inclusion hUV y) …
          g : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          ⊢ Eq (HMul.hMul ((SmoothMap.restrictRingHom IM (modelWithCornersSelf 𝕜 𝕜) 𝕜 hU …
        -/
        apply Subtype.ext
        /-
          case h.e'_2.h.e'_6.a
          𝕜 : Type u
          inst✝⁵ : NontriviallyNormedField 𝕜
          EM : Type u_1
          inst✝⁴ : NormedAddCommGroup EM
          inst✝³ : NormedSpace 𝕜 EM
          HM : Type u_2
          inst✝² : TopologicalSpace HM
          IM : ModelWithCorners 𝕜 EM HM
          M : Type u
          inst✝¹ : TopologicalSpace M
          inst✝ : ChartedSpace HM M
          x : M
          S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
          U : TopologicalSpace.Opens M
          hxU : Membership.mem U x
          f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
          hf' : Ne (f ⟨x, hxU⟩) 0
          V : TopologicalSpace.Opens M
          hUV : LE.le V U
          hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem (Set.range  …
          hV₀ : IsOpen (Set.range (Set.inclusion hUV))
          hxV₀ : Membership.mem (Set.range (Set.inclusion hUV)) ⟨x, hxU⟩
          hxV : Membership.mem (↑V) x
          hVf : ∀ (y : Subtype fun x => Membership.mem V x), Ne (f (Set.inclusion hUV y) …
          g : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          ⊢ Eq ↑(HMul.hMul ((SmoothMap.restrictRingHom IM (modelWithCornersSelf 𝕜 𝕜) 𝕜 h …
        -/
        ext y
        /-
          case h.e'_2.h.e'_6.a.h
          𝕜 : Type u
          inst✝⁵ : NontriviallyNormedField 𝕜
          EM : Type u_1
          inst✝⁴ : NormedAddCommGroup EM
          inst✝³ : NormedSpace 𝕜 EM
          HM : Type u_2
          inst✝² : TopologicalSpace HM
          IM : ModelWithCorners 𝕜 EM HM
          M : Type u
          inst✝¹ : TopologicalSpace M
          inst✝ : ChartedSpace HM M
          x : M
          S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
          U : TopologicalSpace.Opens M
          hxU : Membership.mem U x
          f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
          hf' : Ne (f ⟨x, hxU⟩) 0
          V : TopologicalSpace.Opens M
          hUV : LE.le V U
          hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem (Set.range  …
          hV₀ : IsOpen (Set.range (Set.inclusion hUV))
          hxV₀ : Membership.mem (Set.range (Set.inclusion hUV)) ⟨x, hxU⟩
          hxV : Membership.mem (↑V) x
          hVf : ∀ (y : Subtype fun x => Membership.mem V x), Ne (f (Set.inclusion hUV y) …
          g : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          y : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Eq (↑(HMul.hMul ((SmoothMap.restrictRingHom IM (modelWithCornersSelf 𝕜 𝕜) 𝕜  …
        -/
        apply mul_inv_cancel₀
        /-
          case h.e'_2.h.e'_6.a.h.h
          𝕜 : Type u
          inst✝⁵ : NontriviallyNormedField 𝕜
          EM : Type u_1
          inst✝⁴ : NormedAddCommGroup EM
          inst✝³ : NormedSpace 𝕜 EM
          HM : Type u_2
          inst✝² : TopologicalSpace HM
          IM : ModelWithCorners 𝕜 EM HM
          M : Type u
          inst✝¹ : TopologicalSpace M
          inst✝ : ChartedSpace HM M
          x : M
          S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
          U : TopologicalSpace.Opens M
          hxU : Membership.mem U x
          f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
          hf' : Ne (f ⟨x, hxU⟩) 0
          V : TopologicalSpace.Opens M
          hUV : LE.le V U
          hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem (Set.range  …
          hV₀ : IsOpen (Set.range (Set.inclusion hUV))
          hxV₀ : Membership.mem (Set.range (Set.inclusion hUV)) ⟨x, hxU⟩
          hxV : Membership.mem (↑V) x
          hVf : ∀ (y : Subtype fun x => Membership.mem V x), Ne (f (Set.inclusion hUV y) …
          g : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          y : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Ne (((SmoothMap.restrictRingHom IM (modelWithCornersSelf 𝕜 𝕜) 𝕜 hUV) f) y) 0
        -/
        exact hVf y
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.intro.intro.intro.intro.intro.refine_2.refine_2
          𝕜 : Type u
          inst✝⁵ : NontriviallyNormedField 𝕜
          EM : Type u_1
          inst✝⁴ : NormedAddCommGroup EM
          inst✝³ : NormedSpace 𝕜 EM
          HM : Type u_2
          inst✝² : TopologicalSpace HM
          IM : ModelWithCorners 𝕜 EM HM
          M : Type u
          inst✝¹ : TopologicalSpace M
          inst✝ : ChartedSpace HM M
          x : M
          S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
          U : TopologicalSpace.Opens M
          hxU : Membership.mem U x
          f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
          hf' : Ne (f ⟨x, hxU⟩) 0
          V : TopologicalSpace.Opens M
          hUV : LE.le V U
          hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem (Set.range  …
          hV₀ : IsOpen (Set.range (Set.inclusion hUV))
          hxV₀ : Membership.mem (Set.range (Set.inclusion hUV)) ⟨x, hxU⟩
          hxV : Membership.mem (↑V) x
          hVf : ∀ (y : Subtype fun x => Membership.mem V x), Ne (f (Set.inclusion hUV y) …
          g : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          ⊢ Eq (HMul.hMul ((S.germ V x hxV).hom g) ((S.germ V x hxV).hom ((SmoothMap.res …
        -/
      · rw [← map_mul]
        -- Qualified the name to avoid Lean not finding a `OneHomClass` https://github.com/leanprover-community/mathlib4/pull/8386
        /-
          case mpr.intro.intro.intro.intro.intro.intro.refine_2.refine_2
          𝕜 : Type u
          inst✝⁵ : NontriviallyNormedField 𝕜
          EM : Type u_1
          inst✝⁴ : NormedAddCommGroup EM
          inst✝³ : NormedSpace 𝕜 EM
          HM : Type u_2
          inst✝² : TopologicalSpace HM
          IM : ModelWithCorners 𝕜 EM HM
          M : Type u
          inst✝¹ : TopologicalSpace M
          inst✝ : ChartedSpace HM M
          x : M
          S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
          U : TopologicalSpace.Opens M
          hxU : Membership.mem U x
          f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
          hf' : Ne (f ⟨x, hxU⟩) 0
          V : TopologicalSpace.Opens M
          hUV : LE.le V U
          hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem (Set.range  …
          hV₀ : IsOpen (Set.range (Set.inclusion hUV))
          hxV₀ : Membership.mem (Set.range (Set.inclusion hUV)) ⟨x, hxU⟩
          hxV : Membership.mem (↑V) x
          hVf : ∀ (y : Subtype fun x => Membership.mem V x), Ne (f (Set.inclusion hUV y) …
          g : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          ⊢ Eq ((S.germ V x hxV).hom (HMul.hMul g ((SmoothMap.restrictRingHom IM (modelW …
        -/
        convert RingHom.map_one _
        /-
          case h.e'_2.h.e'_6
          𝕜 : Type u
          inst✝⁵ : NontriviallyNormedField 𝕜
          EM : Type u_1
          inst✝⁴ : NormedAddCommGroup EM
          inst✝³ : NormedSpace 𝕜 EM
          HM : Type u_2
          inst✝² : TopologicalSpace HM
          IM : ModelWithCorners 𝕜 EM HM
          M : Type u
          inst✝¹ : TopologicalSpace M
          inst✝ : ChartedSpace HM M
          x : M
          S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
          U : TopologicalSpace.Opens M
          hxU : Membership.mem U x
          f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
          hf' : Ne (f ⟨x, hxU⟩) 0
          V : TopologicalSpace.Opens M
          hUV : LE.le V U
          hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem (Set.range  …
          hV₀ : IsOpen (Set.range (Set.inclusion hUV))
          hxV₀ : Membership.mem (Set.range (Set.inclusion hUV)) ⟨x, hxU⟩
          hxV : Membership.mem (↑V) x
          hVf : ∀ (y : Subtype fun x => Membership.mem V x), Ne (f (Set.inclusion hUV y) …
          g : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          ⊢ Eq (HMul.hMul g ((SmoothMap.restrictRingHom IM (modelWithCornersSelf 𝕜 𝕜) 𝕜  …
        -/
        apply Subtype.ext
        /-
          case h.e'_2.h.e'_6.a
          𝕜 : Type u
          inst✝⁵ : NontriviallyNormedField 𝕜
          EM : Type u_1
          inst✝⁴ : NormedAddCommGroup EM
          inst✝³ : NormedSpace 𝕜 EM
          HM : Type u_2
          inst✝² : TopologicalSpace HM
          IM : ModelWithCorners 𝕜 EM HM
          M : Type u
          inst✝¹ : TopologicalSpace M
          inst✝ : ChartedSpace HM M
          x : M
          S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
          U : TopologicalSpace.Opens M
          hxU : Membership.mem U x
          f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
          hf' : Ne (f ⟨x, hxU⟩) 0
          V : TopologicalSpace.Opens M
          hUV : LE.le V U
          hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem (Set.range  …
          hV₀ : IsOpen (Set.range (Set.inclusion hUV))
          hxV₀ : Membership.mem (Set.range (Set.inclusion hUV)) ⟨x, hxU⟩
          hxV : Membership.mem (↑V) x
          hVf : ∀ (y : Subtype fun x => Membership.mem V x), Ne (f (Set.inclusion hUV y) …
          g : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          ⊢ Eq ↑(HMul.hMul g ((SmoothMap.restrictRingHom IM (modelWithCornersSelf 𝕜 𝕜) 𝕜 …
        -/
        ext y
        /-
          case h.e'_2.h.e'_6.a.h
          𝕜 : Type u
          inst✝⁵ : NontriviallyNormedField 𝕜
          EM : Type u_1
          inst✝⁴ : NormedAddCommGroup EM
          inst✝³ : NormedSpace 𝕜 EM
          HM : Type u_2
          inst✝² : TopologicalSpace HM
          IM : ModelWithCorners 𝕜 EM HM
          M : Type u
          inst✝¹ : TopologicalSpace M
          inst✝ : ChartedSpace HM M
          x : M
          S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
          U : TopologicalSpace.Opens M
          hxU : Membership.mem U x
          f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
          hf' : Ne (f ⟨x, hxU⟩) 0
          V : TopologicalSpace.Opens M
          hUV : LE.le V U
          hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem (Set.range  …
          hV₀ : IsOpen (Set.range (Set.inclusion hUV))
          hxV₀ : Membership.mem (Set.range (Set.inclusion hUV)) ⟨x, hxU⟩
          hxV : Membership.mem (↑V) x
          hVf : ∀ (y : Subtype fun x => Membership.mem V x), Ne (f (Set.inclusion hUV y) …
          g : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          y : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Eq (↑(HMul.hMul g ((SmoothMap.restrictRingHom IM (modelWithCornersSelf 𝕜 𝕜)  …
        -/
        apply inv_mul_cancel₀
        /-
          case h.e'_2.h.e'_6.a.h.h
          𝕜 : Type u
          inst✝⁵ : NontriviallyNormedField 𝕜
          EM : Type u_1
          inst✝⁴ : NormedAddCommGroup EM
          inst✝³ : NormedSpace 𝕜 EM
          HM : Type u_2
          inst✝² : TopologicalSpace HM
          IM : ModelWithCorners 𝕜 EM HM
          M : Type u
          inst✝¹ : TopologicalSpace M
          inst✝ : ChartedSpace HM M
          x : M
          S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
          U : TopologicalSpace.Opens M
          hxU : Membership.mem U x
          f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
          hf' : Ne (f ⟨x, hxU⟩) 0
          V : TopologicalSpace.Opens M
          hUV : LE.le V U
          hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem (Set.range  …
          hV₀ : IsOpen (Set.range (Set.inclusion hUV))
          hxV₀ : Membership.mem (Set.range (Set.inclusion hUV)) ⟨x, hxU⟩
          hxV : Membership.mem (↑V) x
          hVf : ∀ (y : Subtype fun x => Membership.mem V x), Ne (f (Set.inclusion hUV y) …
          g : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
          y : Subtype fun x => Membership.mem (Opposite.unop { unop := V }) x
          ⊢ Ne (Function.comp (⇑f) (Set.inclusion hUV) y) 0
        -/
        exact hVf y
        /-
          🎉 no goals
        -/
      /-
        case mpr.intro.intro.intro.intro.intro.intro.refine_1
        𝕜 : Type u
        inst✝⁵ : NontriviallyNormedField 𝕜
        EM : Type u_1
        inst✝⁴ : NormedAddCommGroup EM
        inst✝³ : NormedSpace 𝕜 EM
        HM : Type u_2
        inst✝² : TopologicalSpace HM
        IM : ModelWithCorners 𝕜 EM HM
        M : Type u
        inst✝¹ : TopologicalSpace M
        inst✝ : ChartedSpace HM M
        x : M
        S : TopCat.Presheaf CommRingCat (TopCat.of M) := (smoothSheafCommRing IM (mode …
        U : TopologicalSpace.Opens M
        hxU : Membership.mem U x
        f : ContMDiffMap IM (modelWithCornersSelf 𝕜 𝕜) (Subtype fun x => Membership.me …
        hf : Ne ((smoothSheafCommRing.eval IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜 x) ((S.ge …
        hf' : Ne (f ⟨x, hxU⟩) 0
        V : TopologicalSpace.Opens M
        hUV : LE.le V U
        hV₀f : ∀ (y : Subtype fun x => Membership.mem U x), Membership.mem (Set.range  …
        hV₀ : IsOpen (Set.range (Set.inclusion hUV))
        hxV₀ : Membership.mem (Set.range (Set.inclusion hUV)) ⟨x, hxU⟩
        hxV : Membership.mem (↑V) x
        hVf : ∀ (y : Subtype fun x => Membership.mem V x), Ne (f (Set.inclusion hUV y) …
        ⊢ ContMDiff IM (modelWithCornersSelf 𝕜 𝕜) Top.top (Inv.inv (Function.comp (⇑f) …
      -/
    · intro y
      #adaptation_note /-- https://github.com/leanprover/lean4/pull/6024
        was `exact`; somehow `convert` bypasess unification issues -/
      convert ((contDiffAt_inv _ (hVf y)).contMDiffAt).comp y
        (f.contMDiff.comp (contMDiff_inclusion hUV)).contMDiffAt


/-- The non-units of the stalk at `x` of the sheaf of smooth functions from `M` to `𝕜`, considered
as a sheaf of commutative rings, are the functions whose values at `x` are zero. -/
theorem smoothSheafCommRing.nonunits_stalk (x : M) :
    nonunits ((smoothSheafCommRing IM 𝓘(𝕜) M 𝕜).presheaf.stalk x)
    = RingHom.ker (smoothSheafCommRing.eval IM 𝓘(𝕜) M 𝕜 x) := by
  /-
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    EM : Type u_1
    inst✝⁴ : NormedAddCommGroup EM
    inst✝³ : NormedSpace 𝕜 EM
    HM : Type u_2
    inst✝² : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    M : Type u
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace HM M
    x : M
    ⊢ Eq (nonunits ↑((smoothSheafCommRing IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜).presh …
  -/
  ext1 f
  /-
    case h
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    EM : Type u_1
    inst✝⁴ : NormedAddCommGroup EM
    inst✝³ : NormedSpace 𝕜 EM
    HM : Type u_2
    inst✝² : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    M : Type u
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace HM M
    x : M
    f : ↑((smoothSheafCommRing IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜).presheaf.stalk x)
    ⊢ Iff (Membership.mem (nonunits ↑((smoothSheafCommRing IM (modelWithCornersSel …
  -/
  rw [mem_nonunits_iff, not_iff_comm, Iff.comm]
  /-
    case h
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    EM : Type u_1
    inst✝⁴ : NormedAddCommGroup EM
    inst✝³ : NormedSpace 𝕜 EM
    HM : Type u_2
    inst✝² : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    M : Type u
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace HM M
    x : M
    f : ↑((smoothSheafCommRing IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜).presheaf.stalk x)
    ⊢ Iff (IsUnit f) (Not (Membership.mem (↑(RingHom.ker (smoothSheafCommRing.eval …
  -/
  apply smoothSheafCommRing.isUnit_stalk_iff
  /-
    🎉 no goals
  -/


/-- The stalks of the structure sheaf of a smooth manifold-with-corners are local rings. -/
instance smoothSheafCommRing.instLocalRing_stalk (x : M) :
    IsLocalRing ((smoothSheafCommRing IM 𝓘(𝕜) M 𝕜).presheaf.stalk x) := by
  /-
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    EM : Type u_1
    inst✝⁴ : NormedAddCommGroup EM
    inst✝³ : NormedSpace 𝕜 EM
    HM : Type u_2
    inst✝² : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    M : Type u
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace HM M
    x : M
    ⊢ IsLocalRing ↑((smoothSheafCommRing IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜).preshe …
  -/
  apply IsLocalRing.of_nonunits_add
  /-
    case h
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    EM : Type u_1
    inst✝⁴ : NormedAddCommGroup EM
    inst✝³ : NormedSpace 𝕜 EM
    HM : Type u_2
    inst✝² : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    M : Type u
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace HM M
    x : M
    ⊢ ∀ (a b : ↑((smoothSheafCommRing IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜).presheaf. …
  -/
  rw [smoothSheafCommRing.nonunits_stalk]
  /-
    case h
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    EM : Type u_1
    inst✝⁴ : NormedAddCommGroup EM
    inst✝³ : NormedSpace 𝕜 EM
    HM : Type u_2
    inst✝² : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    M : Type u
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace HM M
    x : M
    ⊢ ∀ (a b : ↑((smoothSheafCommRing IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜).presheaf. …
  -/
  intro f g
  /-
    case h
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    EM : Type u_1
    inst✝⁴ : NormedAddCommGroup EM
    inst✝³ : NormedSpace 𝕜 EM
    HM : Type u_2
    inst✝² : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    M : Type u
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace HM M
    x : M
    f g : ↑((smoothSheafCommRing IM (modelWithCornersSelf 𝕜 𝕜) M 𝕜).presheaf.stalk …
    ⊢ Membership.mem (↑(RingHom.ker (smoothSheafCommRing.eval IM (modelWithCorners …
  -/
  exact Ideal.add_mem _
  /-
    🎉 no goals
  -/


/-- A smooth manifold-with-corners can be considered as a locally ringed space. -/
def SmoothManifoldWithCorners.locallyRingedSpace : LocallyRingedSpace where
  carrier := TopCat.of M
  presheaf := smoothPresheafCommRing IM 𝓘(𝕜) M 𝕜
  IsSheaf := (smoothSheafCommRing IM 𝓘(𝕜) M 𝕜).cond
  isLocalRing x := smoothSheafCommRing.instLocalRing_stalk IM x

