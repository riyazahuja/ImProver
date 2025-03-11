variable (I I') in
/-- Property in the model space of a model with corners of being `C^n` within at set at a point,
when read in the model vector space. This property will be lifted to manifolds to define smooth
functions between manifolds. -/
def ContDiffWithinAtProp (n : ℕ∞) (f : H → H') (s : Set H) (x : H) : Prop :=
  ContDiffWithinAt 𝕜 n (I' ∘ f ∘ I.symm) (I.symm ⁻¹' s ∩ range I) (I x)


theorem contDiffWithinAtProp_self_source {f : E → H'} {s : Set E} {x : E} :
    ContDiffWithinAtProp 𝓘(𝕜, E) I' n f s x ↔ ContDiffWithinAt 𝕜 n (I' ∘ f) s x := by
  simp_rw [ContDiffWithinAtProp, modelWithCornersSelf_coe, range_id, inter_univ,
    modelWithCornersSelf_coe_symm, CompTriple.comp_eq, preimage_id_eq, id_eq]


theorem contDiffWithinAtProp_self {f : E → E'} {s : Set E} {x : E} :
    ContDiffWithinAtProp 𝓘(𝕜, E) 𝓘(𝕜, E') n f s x ↔ ContDiffWithinAt 𝕜 n f s x :=
  contDiffWithinAtProp_self_source


theorem contDiffWithinAtProp_self_target {f : H → E'} {s : Set H} {x : H} :
    ContDiffWithinAtProp I 𝓘(𝕜, E') n f s x ↔
      ContDiffWithinAt 𝕜 n (f ∘ I.symm) (I.symm ⁻¹' s ∩ range I) (I x) :=
  Iff.rfl


/-- Being `Cⁿ` in the model space is a local property, invariant under smooth maps. Therefore,
it will lift nicely to manifolds. -/
theorem contDiffWithinAt_localInvariantProp (n : ℕ∞) :
    (contDiffGroupoid ∞ I).LocalInvariantProp (contDiffGroupoid ∞ I')
      (ContDiffWithinAtProp I I' n) where
  is_local {s x u f} u_open xu := by
    have : I.symm ⁻¹' (s ∩ u) ∩ range I = I.symm ⁻¹' s ∩ range I ∩ I.symm ⁻¹' u := by
      simp only [inter_right_comm, preimage_inter]
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      E' : Type u_5
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      n : ENat
      s : Set H
      x : H
      u : Set H
      f : H → H'
      u_open : IsOpen u
      xu : Membership.mem u x
      this : Eq (Inter.inter (Set.preimage (↑I.symm) (Inter.inter s u)) (Set.range ↑ …
      ⊢ Iff (ContDiffWithinAtProp I I' n f s x) (ContDiffWithinAtProp I I' n f (Inte …
    -/
    rw [ContDiffWithinAtProp, ContDiffWithinAtProp, this]
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      E' : Type u_5
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      n : ENat
      s : Set H
      x : H
      u : Set H
      f : H → H'
      u_open : IsOpen u
      xu : Membership.mem u x
      this : Eq (Inter.inter (Set.preimage (↑I.symm) (Inter.inter s u)) (Set.range ↑ …
      ⊢ Iff (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp f ↑I.symm)) …
    -/
    symm
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      E' : Type u_5
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      n : ENat
      s : Set H
      x : H
      u : Set H
      f : H → H'
      u_open : IsOpen u
      xu : Membership.mem u x
      this : Eq (Inter.inter (Set.preimage (↑I.symm) (Inter.inter s u)) (Set.range ↑ …
      ⊢ Iff (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp f ↑I.symm)) …
    -/
    apply contDiffWithinAt_inter
    have : u ∈ 𝓝 (I.symm (I x)) := by
      rw [ModelWithCorners.left_inv]
      exact u_open.mem_nhds xu
    /-
      case h
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      E' : Type u_5
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      n : ENat
      s : Set H
      x : H
      u : Set H
      f : H → H'
      u_open : IsOpen u
      xu : Membership.mem u x
      this✝ : Eq (Inter.inter (Set.preimage (↑I.symm) (Inter.inter s u)) (Set.range  …
      this : Membership.mem (nhds (↑I.symm (↑I x))) u
      ⊢ Membership.mem (nhds (↑I x)) (Set.preimage (↑I.symm) u)
    -/
    apply ContinuousAt.preimage_mem_nhds I.continuous_symm.continuousAt this
    /-
      🎉 no goals
    -/
  right_invariance' {s x f e} he hx h := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      E' : Type u_5
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      n : ENat
      s : Set H
      x : H
      f : H → H'
      e : PartialHomeomorph H H
      he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
      hx : Membership.mem e.source x
      h : ContDiffWithinAtProp I I' n f s x
      ⊢ ContDiffWithinAtProp I I' n (Function.comp f ↑e.symm) (Set.preimage (↑e.symm …
    -/
    rw [ContDiffWithinAtProp] at h ⊢
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      E' : Type u_5
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      n : ENat
      s : Set H
      x : H
      f : H → H'
      e : PartialHomeomorph H H
      he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
      hx : Membership.mem e.source x
      h : ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp f ↑I.symm)) (I …
      ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp (Function.comp f …
    -/
    have : I x = (I ∘ e.symm ∘ I.symm) (I (e x)) := by simp only [hx, mfld_simps]
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      E' : Type u_5
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      n : ENat
      s : Set H
      x : H
      f : H → H'
      e : PartialHomeomorph H H
      he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
      hx : Membership.mem e.source x
      h : ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp f ↑I.symm)) (I …
      this : Eq (↑I x) (Function.comp (↑I) (Function.comp ↑e.symm ↑I.symm) (↑I (↑e x …
      ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp (Function.comp f …
    -/
    rw [this] at h
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      E' : Type u_5
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      n : ENat
      s : Set H
      x : H
      f : H → H'
      e : PartialHomeomorph H H
      he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
      hx : Membership.mem e.source x
      h : ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp f ↑I.symm)) (I …
      this : Eq (↑I x) (Function.comp (↑I) (Function.comp ↑e.symm ↑I.symm) (↑I (↑e x …
      ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp (Function.comp f …
    -/
    have : I (e x) ∈ I.symm ⁻¹' e.target ∩ range I := by simp only [hx, mfld_simps]
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      E' : Type u_5
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      n : ENat
      s : Set H
      x : H
      f : H → H'
      e : PartialHomeomorph H H
      he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
      hx : Membership.mem e.source x
      h : ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp f ↑I.symm)) (I …
      this✝ : Eq (↑I x) (Function.comp (↑I) (Function.comp ↑e.symm ↑I.symm) (↑I (↑e  …
      this : Membership.mem (Inter.inter (Set.preimage (↑I.symm) e.target) (Set.rang …
      ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp (Function.comp f …
    -/
    have := (mem_groupoid_of_pregroupoid.2 he).2.contDiffWithinAt this
    convert (h.comp_inter _ (this.of_le (mod_cast le_top))).mono_of_mem_nhdsWithin _
      using 1
      /-
        case h.e'_10
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝³ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        E' : Type u_5
        inst✝² : NormedAddCommGroup E'
        inst✝¹ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        n : ENat
        s : Set H
        x : H
        f : H → H'
        e : PartialHomeomorph H H
        he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
        hx : Membership.mem e.source x
        h : ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp f ↑I.symm)) (I …
        this✝¹ : Eq (↑I x) (Function.comp (↑I) (Function.comp ↑e.symm ↑I.symm) (↑I (↑e …
        this✝ : Membership.mem (Inter.inter (Set.preimage (↑I.symm) e.target) (Set.ran …
        this : ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑e.sym …
        ⊢ Eq (Function.comp (↑I') (Function.comp (Function.comp f ↑e.symm) ↑I.symm)) ( …
      -/
    · ext y; simp only [mfld_simps]
             /-
               🎉 no goals
             -/
    refine mem_nhdsWithin.mpr
      ⟨I.symm ⁻¹' e.target, e.open_target.preimage I.continuous_symm, by
        simp_rw [mem_preimage, I.left_inv, e.mapsTo hx], ?_⟩
    /-
      case convert_2
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      E' : Type u_5
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      n : ENat
      s : Set H
      x : H
      f : H → H'
      e : PartialHomeomorph H H
      he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
      hx : Membership.mem e.source x
      h : ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp f ↑I.symm)) (I …
      this✝¹ : Eq (↑I x) (Function.comp (↑I) (Function.comp ↑e.symm ↑I.symm) (↑I (↑e …
      this✝ : Membership.mem (Inter.inter (Set.preimage (↑I.symm) e.target) (Set.ran …
      this : ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑e.sym …
      ⊢ HasSubset.Subset (Inter.inter (Set.preimage (↑I.symm) e.target) (Inter.inter …
    -/
    mfld_set_tac
    /-
      🎉 no goals
    -/
  congr_of_forall {s x f g} h hx hf := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      E' : Type u_5
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      n : ENat
      s : Set H
      x : H
      f g : H → H'
      h : ∀ (y : H), Membership.mem s y → Eq (f y) (g y)
      hx : Eq (f x) (g x)
      hf : ContDiffWithinAtProp I I' n f s x
      ⊢ ContDiffWithinAtProp I I' n g s x
    -/
    apply hf.congr
      /-
        case h₁
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝³ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        E' : Type u_5
        inst✝² : NormedAddCommGroup E'
        inst✝¹ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        n : ENat
        s : Set H
        x : H
        f g : H → H'
        h : ∀ (y : H), Membership.mem s y → Eq (f y) (g y)
        hx : Eq (f x) (g x)
        hf : ContDiffWithinAtProp I I' n f s x
        ⊢ ∀ (y : E), Membership.mem (Inter.inter (Set.preimage (↑I.symm) s) (Set.range …
      -/
    · intro y hy
      /-
        case h₁
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝³ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        E' : Type u_5
        inst✝² : NormedAddCommGroup E'
        inst✝¹ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        n : ENat
        s : Set H
        x : H
        f g : H → H'
        h : ∀ (y : H), Membership.mem s y → Eq (f y) (g y)
        hx : Eq (f x) (g x)
        hf : ContDiffWithinAtProp I I' n f s x
        y : E
        hy : Membership.mem (Inter.inter (Set.preimage (↑I.symm) s) (Set.range ↑I)) y
        ⊢ Eq (Function.comp (↑I') (Function.comp g ↑I.symm) y) (Function.comp (↑I') (F …
      -/
      simp only [mfld_simps] at hy
      /-
        case h₁
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝³ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        E' : Type u_5
        inst✝² : NormedAddCommGroup E'
        inst✝¹ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        n : ENat
        s : Set H
        x : H
        f g : H → H'
        h : ∀ (y : H), Membership.mem s y → Eq (f y) (g y)
        hx : Eq (f x) (g x)
        hf : ContDiffWithinAtProp I I' n f s x
        y : E
        hy : And (Membership.mem s (↑I.symm y)) (Membership.mem (Set.range ↑I) y)
        ⊢ Eq (Function.comp (↑I') (Function.comp g ↑I.symm) y) (Function.comp (↑I') (F …
      -/
      simp only [h, hy, mfld_simps]
      /-
        🎉 no goals
      -/
      /-
        case hx
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝³ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        E' : Type u_5
        inst✝² : NormedAddCommGroup E'
        inst✝¹ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        n : ENat
        s : Set H
        x : H
        f g : H → H'
        h : ∀ (y : H), Membership.mem s y → Eq (f y) (g y)
        hx : Eq (f x) (g x)
        hf : ContDiffWithinAtProp I I' n f s x
        ⊢ Eq (Function.comp (↑I') (Function.comp g ↑I.symm) (↑I x)) (Function.comp (↑I …
      -/
    · simp only [hx, mfld_simps]
      /-
        🎉 no goals
      -/
  left_invariance' {s x f e'} he' hs hx h := by
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      E' : Type u_5
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      n : ENat
      s : Set H
      x : H
      f : H → H'
      e' : PartialHomeomorph H' H'
      he' : Membership.mem (contDiffGroupoid (↑Top.top) I') e'
      hs : HasSubset.Subset s (Set.preimage f e'.source)
      hx : Membership.mem e'.source (f x)
      h : ContDiffWithinAtProp I I' n f s x
      ⊢ ContDiffWithinAtProp I I' n (Function.comp (↑e') f) s x
    -/
    rw [ContDiffWithinAtProp] at h ⊢
    have A : (I' ∘ f ∘ I.symm) (I x) ∈ I'.symm ⁻¹' e'.source ∩ range I' := by
      simp only [hx, mfld_simps]
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      E' : Type u_5
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      n : ENat
      s : Set H
      x : H
      f : H → H'
      e' : PartialHomeomorph H' H'
      he' : Membership.mem (contDiffGroupoid (↑Top.top) I') e'
      hs : HasSubset.Subset s (Set.preimage f e'.source)
      hx : Membership.mem e'.source (f x)
      h : ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp f ↑I.symm)) (I …
      A : Membership.mem (Inter.inter (Set.preimage (↑I'.symm) e'.source) (Set.range …
      ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp (Function.comp ( …
    -/
    have := (mem_groupoid_of_pregroupoid.2 he').1.contDiffWithinAt A
    /-
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝³ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      E' : Type u_5
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      n : ENat
      s : Set H
      x : H
      f : H → H'
      e' : PartialHomeomorph H' H'
      he' : Membership.mem (contDiffGroupoid (↑Top.top) I') e'
      hs : HasSubset.Subset s (Set.preimage f e'.source)
      hx : Membership.mem e'.source (f x)
      h : ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp f ↑I.symm)) (I …
      A : Membership.mem (Inter.inter (Set.preimage (↑I'.symm) e'.source) (Set.range …
      this : ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I') (Function.comp ↑e' ↑ …
      ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp (Function.comp ( …
    -/
    convert (this.of_le (mod_cast le_top)).comp _ h _
      /-
        case h.e'_10
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝³ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        E' : Type u_5
        inst✝² : NormedAddCommGroup E'
        inst✝¹ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        n : ENat
        s : Set H
        x : H
        f : H → H'
        e' : PartialHomeomorph H' H'
        he' : Membership.mem (contDiffGroupoid (↑Top.top) I') e'
        hs : HasSubset.Subset s (Set.preimage f e'.source)
        hx : Membership.mem e'.source (f x)
        h : ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp f ↑I.symm)) (I …
        A : Membership.mem (Inter.inter (Set.preimage (↑I'.symm) e'.source) (Set.range …
        this : ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I') (Function.comp ↑e' ↑ …
        ⊢ Eq (Function.comp (↑I') (Function.comp (Function.comp (↑e') f) ↑I.symm)) (Fu …
      -/
    · ext y; simp only [mfld_simps]
             /-
               🎉 no goals
             -/
      /-
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝³ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        E' : Type u_5
        inst✝² : NormedAddCommGroup E'
        inst✝¹ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        n : ENat
        s : Set H
        x : H
        f : H → H'
        e' : PartialHomeomorph H' H'
        he' : Membership.mem (contDiffGroupoid (↑Top.top) I') e'
        hs : HasSubset.Subset s (Set.preimage f e'.source)
        hx : Membership.mem e'.source (f x)
        h : ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp f ↑I.symm)) (I …
        A : Membership.mem (Inter.inter (Set.preimage (↑I'.symm) e'.source) (Set.range …
        this : ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I') (Function.comp ↑e' ↑ …
        ⊢ Set.MapsTo (Function.comp (↑I') (Function.comp f ↑I.symm)) (Inter.inter (Set …
      -/
    · intro y hy; simp only [mfld_simps] at hy; simpa only [hy, mfld_simps] using hs hy.1
                                                /-
                                                  🎉 no goals
                                                -/


theorem contDiffWithinAtProp_mono_of_mem_nhdsWithin
    (n : ℕ∞) ⦃s x t⦄ ⦃f : H → H'⦄ (hts : s ∈ 𝓝[t] x)
    (h : ContDiffWithinAtProp I I' n f s x) : ContDiffWithinAtProp I I' n f t x := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    E' : Type u_5
    inst✝² : NormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    n : ENat
    s : Set H
    x : H
    t : Set H
    f : H → H'
    hts : Membership.mem (nhdsWithin x t) s
    h : ContDiffWithinAtProp I I' n f s x
    ⊢ ContDiffWithinAtProp I I' n f t x
  -/
  refine h.mono_of_mem_nhdsWithin ?_
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    E' : Type u_5
    inst✝² : NormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    n : ENat
    s : Set H
    x : H
    t : Set H
    f : H → H'
    hts : Membership.mem (nhdsWithin x t) s
    h : ContDiffWithinAtProp I I' n f s x
    ⊢ Membership.mem (nhdsWithin (↑I x) (Inter.inter (Set.preimage (↑I.symm) t) (S …
  -/
  refine inter_mem ?_ (mem_of_superset self_mem_nhdsWithin inter_subset_right)
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    E' : Type u_5
    inst✝² : NormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    n : ENat
    s : Set H
    x : H
    t : Set H
    f : H → H'
    hts : Membership.mem (nhdsWithin x t) s
    h : ContDiffWithinAtProp I I' n f s x
    ⊢ Membership.mem (nhdsWithin (↑I x) (Inter.inter (Set.preimage (↑I.symm) t) (S …
  -/
  rwa [← Filter.mem_map, ← I.image_eq, I.symm_map_nhdsWithin_image]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-31")]
alias contDiffWithinAtProp_mono_of_mem := contDiffWithinAtProp_mono_of_mem_nhdsWithin


theorem contDiffWithinAtProp_id (x : H) : ContDiffWithinAtProp I I n id univ x := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x : H
    ⊢ ContDiffWithinAtProp I I n id Set.univ x
  -/
  simp only [ContDiffWithinAtProp, id_comp, preimage_univ, univ_inter]
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x : H
    ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp ↑I ↑I.symm) (Set.range ↑I) (↑I x)
  -/
  have : ContDiffWithinAt 𝕜 n id (range I) (I x) := contDiff_id.contDiffAt.contDiffWithinAt
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    n : ENat
    x : H
    this : ContDiffWithinAt 𝕜 (↑n) id (Set.range ↑I) (↑I x)
    ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp ↑I ↑I.symm) (Set.range ↑I) (↑I x)
  -/
  refine this.congr (fun y hy => ?_) ?_
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      n : ENat
      x : H
      this : ContDiffWithinAt 𝕜 (↑n) id (Set.range ↑I) (↑I x)
      y : E
      hy : Membership.mem (Set.range ↑I) y
      ⊢ Eq (Function.comp (↑I) (↑I.symm) y) (id y)
    -/
  · simp only [ModelWithCorners.right_inv I hy, mfld_simps]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      n : ENat
      x : H
      this : ContDiffWithinAt 𝕜 (↑n) id (Set.range ↑I) (↑I x)
      ⊢ Eq (Function.comp (↑I) (↑I.symm) (↑I x)) (id (↑I x))
    -/
  · simp only [mfld_simps]
    /-
      🎉 no goals
    -/


variable (I I') in
/-- A function is `n` times continuously differentiable within a set at a point in a manifold if
it is continuous and it is `n` times continuously differentiable in this set around this point, when
read in the preferred chart at this point. -/
def ContMDiffWithinAt (n : ℕ∞) (f : M → M') (s : Set M) (x : M) :=
  LiftPropWithinAt (ContDiffWithinAtProp I I' n) f s x


@[deprecated (since := "024-11-21")] alias SmoothWithinAt := ContMDiffWithinAt


variable (I I') in
/-- A function is `n` times continuously differentiable at a point in a manifold if
it is continuous and it is `n` times continuously differentiable around this point, when
read in the preferred chart at this point. -/
def ContMDiffAt (n : ℕ∞) (f : M → M') (x : M) :=
  ContMDiffWithinAt I I' n f univ x


theorem contMDiffAt_iff {n : ℕ∞} {f : M → M'} {x : M} :
    ContMDiffAt I I' n f x ↔
      ContinuousAt f x ∧
        ContDiffWithinAt 𝕜 n (extChartAt I' (f x) ∘ f ∘ (extChartAt I x).symm) (range I)
          (extChartAt I x x) :=
                             /-
                               𝕜 : Type u_1
                               inst✝¹⁰ : NontriviallyNormedField 𝕜
                               E : Type u_2
                               inst✝⁹ : NormedAddCommGroup E
                               inst✝⁸ : NormedSpace 𝕜 E
                               H : Type u_3
                               inst✝⁷ : TopologicalSpace H
                               I : ModelWithCorners 𝕜 E H
                               M : Type u_4
                               inst✝⁶ : TopologicalSpace M
                               inst✝⁵ : ChartedSpace H M
                               E' : Type u_5
                               inst✝⁴ : NormedAddCommGroup E'
                               inst✝³ : NormedSpace 𝕜 E'
                               H' : Type u_6
                               inst✝² : TopologicalSpace H'
                               I' : ModelWithCorners 𝕜 E' H'
                               M' : Type u_7
                               inst✝¹ : TopologicalSpace M'
                               inst✝ : ChartedSpace H' M'
                               n : ENat
                               f : M → M'
                               x : M
                               ⊢ Iff (And (ContinuousAt f x) (ContDiffWithinAtProp I I' n (Function.comp (↑(c …
                             -/
  liftPropAt_iff.trans <| by rw [ContDiffWithinAtProp, preimage_univ, univ_inter]; rfl
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[deprecated (since := "024-11-21")] alias SmoothAt := ContMDiffAt


variable (I I') in
/-- A function is `n` times continuously differentiable in a set of a manifold if it is continuous
and, for any pair of points, it is `n` times continuously differentiable on this set in the charts
around these points. -/
def ContMDiffOn (n : ℕ∞) (f : M → M') (s : Set M) :=
  ∀ x ∈ s, ContMDiffWithinAt I I' n f s x


@[deprecated (since := "024-11-21")] alias SmoothOn := ContMDiffOn


variable (I I') in
/-- A function is `n` times continuously differentiable in a manifold if it is continuous
and, for any pair of points, it is `n` times continuously differentiable in the charts
around these points. -/
def ContMDiff (n : ℕ∞) (f : M → M') :=
  ∀ x, ContMDiffAt I I' n f x


@[deprecated (since := "024-11-21")] alias Smooth := ContMDiff



theorem ContMDiffWithinAt.of_le (hf : ContMDiffWithinAt I I' n f s x) (le : m ≤ n) :
    ContMDiffWithinAt I I' m f s x := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    m n : ENat
    hf : ContMDiffWithinAt I I' n f s x
    le : LE.le m n
    ⊢ ContMDiffWithinAt I I' m f s x
  -/
  simp only [ContMDiffWithinAt, LiftPropWithinAt] at hf ⊢
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    m n : ENat
    hf : ChartedSpace.LiftPropWithinAt (ContDiffWithinAtProp I I' n) f s x
    le : LE.le m n
    ⊢ ChartedSpace.LiftPropWithinAt (ContDiffWithinAtProp I I' m) f s x
  -/
  exact ⟨hf.1, hf.2.of_le (mod_cast le)⟩
  /-
    🎉 no goals
  -/


theorem ContMDiffAt.of_le (hf : ContMDiffAt I I' n f x) (le : m ≤ n) : ContMDiffAt I I' m f x :=
  ContMDiffWithinAt.of_le hf le


theorem ContMDiffOn.of_le (hf : ContMDiffOn I I' n f s) (le : m ≤ n) : ContMDiffOn I I' m f s :=
  fun x hx => (hf x hx).of_le le


theorem ContMDiff.of_le (hf : ContMDiff I I' n f) (le : m ≤ n) : ContMDiff I I' m f := fun x =>
  (hf x).of_le le


@[deprecated (since := "2024-11-20")] alias ContMDiff.smooth := ContMDiff.of_le


@[deprecated (since := "2024-11-20")] alias Smooth.contMDiff := ContMDiff.of_le


@[deprecated (since := "2024-11-20")] alias ContMDiffOn.smoothOn := ContMDiffOn.of_le


@[deprecated (since := "2024-11-20")] alias SmoothOn.contMDiffOn := ContMDiffOn.of_le


@[deprecated (since := "2024-11-20")] alias ContMDiffAt.smoothAt := ContMDiffAt.of_le


@[deprecated (since := "2024-11-20")] alias SmoothAt.contMDiffAt := ContMDiffOn.of_le


@[deprecated (since := "2024-11-20")]
alias ContMDiffWithinAt.smoothWithinAt := ContMDiffWithinAt.of_le


@[deprecated (since := "2024-11-20")]
alias SmoothWithinAt.contMDiffWithinAt := ContMDiffWithinAt.of_le


theorem ContMDiff.contMDiffAt (h : ContMDiff I I' n f) : ContMDiffAt I I' n f x :=
  h x


@[deprecated (since := "2024-11-20")] alias Smooth.smoothAt := ContMDiff.contMDiffAt


theorem contMDiffWithinAt_univ : ContMDiffWithinAt I I' n f univ x ↔ ContMDiffAt I I' n f x :=
  Iff.rfl


@[deprecated (since := "2024-11-20")] alias smoothWithinAt_univ := contMDiffWithinAt_univ


theorem contMDiffOn_univ : ContMDiffOn I I' n f univ ↔ ContMDiff I I' n f := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    f : M → M'
    n : ENat
    ⊢ Iff (ContMDiffOn I I' n f Set.univ) (ContMDiff I I' n f)
  -/
  simp only [ContMDiffOn, ContMDiff, contMDiffWithinAt_univ, forall_prop_of_true, mem_univ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-20")] alias smoothOn_univ := contMDiffOn_univ


/-- One can reformulate smoothness within a set at a point as continuity within this set at this
point, and smoothness in the corresponding extended chart. -/
theorem contMDiffWithinAt_iff :
    ContMDiffWithinAt I I' n f s x ↔
      ContinuousWithinAt f s x ∧
        ContDiffWithinAt 𝕜 n (extChartAt I' (f x) ∘ f ∘ (extChartAt I x).symm)
          ((extChartAt I x).symm ⁻¹' s ∩ range I) (extChartAt I x x) := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    ⊢ Iff (ContMDiffWithinAt I I' n f s x) (And (ContinuousWithinAt f s x) (ContDi …
  -/
  simp_rw [ContMDiffWithinAt, liftPropWithinAt_iff']; rfl
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- One can reformulate smoothness within a set at a point as continuity within this set at this
point, and smoothness in the corresponding extended chart. This form states smoothness of `f`
written in such a way that the set is restricted to lie within the domain/codomain of the
corresponding charts.
Even though this expression is more complicated than the one in `contMDiffWithinAt_iff`, it is
a smaller set, but their germs at `extChartAt I x x` are equal. It is sometimes useful to rewrite
using this in the goal.
-/
theorem contMDiffWithinAt_iff' :
    ContMDiffWithinAt I I' n f s x ↔
      ContinuousWithinAt f s x ∧
        ContDiffWithinAt 𝕜 n (extChartAt I' (f x) ∘ f ∘ (extChartAt I x).symm)
          ((extChartAt I x).target ∩
            (extChartAt I x).symm ⁻¹' (s ∩ f ⁻¹' (extChartAt I' (f x)).source))
          (extChartAt I x x) := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    ⊢ Iff (ContMDiffWithinAt I I' n f s x) (And (ContinuousWithinAt f s x) (ContDi …
  -/
  simp only [ContMDiffWithinAt, liftPropWithinAt_iff']
  exact and_congr_right fun hc => contDiffWithinAt_congr_set <|
    hc.extChartAt_symm_preimage_inter_range_eventuallyEq


/-- One can reformulate smoothness within a set at a point as continuity within this set at this
point, and smoothness in the corresponding extended chart in the target. -/
theorem contMDiffWithinAt_iff_target :
    ContMDiffWithinAt I I' n f s x ↔
      ContinuousWithinAt f s x ∧ ContMDiffWithinAt I 𝓘(𝕜, E') n (extChartAt I' (f x) ∘ f) s x := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    ⊢ Iff (ContMDiffWithinAt I I' n f s x) (And (ContinuousWithinAt f s x) (ContMD …
  -/
  simp_rw [ContMDiffWithinAt, liftPropWithinAt_iff', ← and_assoc]
  have cont :
    ContinuousWithinAt f s x ∧ ContinuousWithinAt (extChartAt I' (f x) ∘ f) s x ↔
        ContinuousWithinAt f s x :=
      and_iff_left_of_imp <| (continuousAt_extChartAt _).comp_continuousWithinAt
  simp_rw [cont, ContDiffWithinAtProp, extChartAt, PartialHomeomorph.extend, PartialEquiv.coe_trans,
    ModelWithCorners.toPartialEquiv_coe, PartialHomeomorph.coe_coe, modelWithCornersSelf_coe,
    chartAt_self_eq, PartialHomeomorph.refl_apply, id_comp]
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    cont : Iff (And (ContinuousWithinAt f s x) (ContinuousWithinAt (Function.comp  …
    ⊢ Iff (And (ContinuousWithinAt f s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-20")] alias smoothWithinAt_iff := contMDiffWithinAt_iff


@[deprecated (since := "2024-11-20")]
alias smoothWithinAt_iff_target := contMDiffWithinAt_iff_target


theorem contMDiffAt_iff_target {x : M} :
    ContMDiffAt I I' n f x ↔
      ContinuousAt f x ∧ ContMDiffAt I 𝓘(𝕜, E') n (extChartAt I' (f x) ∘ f) x := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    f : M → M'
    n : ENat
    x : M
    ⊢ Iff (ContMDiffAt I I' n f x) (And (ContinuousAt f x) (ContMDiffAt I (modelWi …
  -/
  rw [ContMDiffAt, ContMDiffAt, contMDiffWithinAt_iff_target, continuousWithinAt_univ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-20")] alias smoothAt_iff_target := contMDiffAt_iff_target



theorem contMDiffWithinAt_iff_source_of_mem_maximalAtlas
    [SmoothManifoldWithCorners I M] (he : e ∈ maximalAtlas I M) (hx : x ∈ e.source) :
    ContMDiffWithinAt I I' n f s x ↔
      ContMDiffWithinAt 𝓘(𝕜, E) I' n (f ∘ (e.extend I).symm) ((e.extend I).symm ⁻¹' s ∩ range I)
        (e.extend I x) := by
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    e : PartialHomeomorph M H
    f : M → M'
    s : Set M
    x : M
    n : ENat
    inst✝ : SmoothManifoldWithCorners I M
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    hx : Membership.mem e.source x
    ⊢ Iff (ContMDiffWithinAt I I' n f s x) (ContMDiffWithinAt (modelWithCornersSel …
  -/
  have h2x := hx; rw [← e.extend_source (I := I)] at h2x
  simp_rw [ContMDiffWithinAt,
    (contDiffWithinAt_localInvariantProp n).liftPropWithinAt_indep_chart_source he hx,
    StructureGroupoid.liftPropWithinAt_self_source,
    e.extend_symm_continuousWithinAt_comp_right_iff, contDiffWithinAtProp_self_source,
    ContDiffWithinAtProp, Function.comp, e.left_inv hx, (e.extend I).left_inv h2x]
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    e : PartialHomeomorph M H
    f : M → M'
    s : Set M
    x : M
    n : ENat
    inst✝ : SmoothManifoldWithCorners I M
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    hx : Membership.mem e.source x
    h2x : Membership.mem (e.extend I).source x
    ⊢ Iff (And (ContinuousWithinAt (Function.comp f ↑e.symm) (Set.preimage (↑e.sym …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem contMDiffWithinAt_iff_source_of_mem_source
    [SmoothManifoldWithCorners I M] {x' : M} (hx' : x' ∈ (chartAt H x).source) :
    ContMDiffWithinAt I I' n f s x' ↔
      ContMDiffWithinAt 𝓘(𝕜, E) I' n (f ∘ (extChartAt I x).symm)
        ((extChartAt I x).symm ⁻¹' s ∩ range I) (extChartAt I x x') :=
  contMDiffWithinAt_iff_source_of_mem_maximalAtlas (chart_mem_maximalAtlas x) hx'


theorem contMDiffAt_iff_source_of_mem_source
    [SmoothManifoldWithCorners I M] {x' : M} (hx' : x' ∈ (chartAt H x).source) :
    ContMDiffAt I I' n f x' ↔
      ContMDiffWithinAt 𝓘(𝕜, E) I' n (f ∘ (extChartAt I x).symm) (range I) (extChartAt I x x') := by
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    x : M
    n : ENat
    inst✝ : SmoothManifoldWithCorners I M
    x' : M
    hx' : Membership.mem (chartAt H x).source x'
    ⊢ Iff (ContMDiffAt I I' n f x') (ContMDiffWithinAt (modelWithCornersSelf 𝕜 E)  …
  -/
  simp_rw [ContMDiffAt, contMDiffWithinAt_iff_source_of_mem_source hx', preimage_univ, univ_inter]
  /-
    🎉 no goals
  -/


theorem contMDiffWithinAt_iff_target_of_mem_source
    [SmoothManifoldWithCorners I' M'] {x : M} {y : M'} (hy : f x ∈ (chartAt H' y).source) :
    ContMDiffWithinAt I I' n f s x ↔
      ContinuousWithinAt f s x ∧ ContMDiffWithinAt I 𝓘(𝕜, E') n (extChartAt I' y ∘ f) s x := by
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    n : ENat
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hy : Membership.mem (chartAt H' y).source (f x)
    ⊢ Iff (ContMDiffWithinAt I I' n f s x) (And (ContinuousWithinAt f s x) (ContMD …
  -/
  simp_rw [ContMDiffWithinAt]
  rw [(contDiffWithinAt_localInvariantProp n).liftPropWithinAt_indep_chart_target
      (chart_mem_maximalAtlas y) hy,
    and_congr_right]
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    n : ENat
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hy : Membership.mem (chartAt H' y).source (f x)
    ⊢ ContinuousWithinAt f s x → Iff (ChartedSpace.LiftPropWithinAt (ContDiffWithi …
  -/
  intro hf
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    n : ENat
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hy : Membership.mem (chartAt H' y).source (f x)
    hf : ContinuousWithinAt f s x
    ⊢ Iff (ChartedSpace.LiftPropWithinAt (ContDiffWithinAtProp I I' n) (Function.c …
  -/
  simp_rw [StructureGroupoid.liftPropWithinAt_self_target]
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    n : ENat
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hy : Membership.mem (chartAt H' y).source (f x)
    hf : ContinuousWithinAt f s x
    ⊢ Iff (And (ContinuousWithinAt (Function.comp (↑(chartAt H' y)) f) s x) (ContD …
  -/
  simp_rw [((chartAt H' y).continuousAt hy).comp_continuousWithinAt hf]
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    n : ENat
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hy : Membership.mem (chartAt H' y).source (f x)
    hf : ContinuousWithinAt f s x
    ⊢ Iff (And True (ContDiffWithinAtProp I I' n (Function.comp (Function.comp (↑( …
  -/
  rw [← extChartAt_source (I := I')] at hy
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    n : ENat
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hy : Membership.mem (extChartAt I' y).source (f x)
    hf : ContinuousWithinAt f s x
    ⊢ Iff (And True (ContDiffWithinAtProp I I' n (Function.comp (Function.comp (↑( …
  -/
  simp_rw [(continuousAt_extChartAt' hy).comp_continuousWithinAt hf]
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    n : ENat
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hy : Membership.mem (extChartAt I' y).source (f x)
    hf : ContinuousWithinAt f s x
    ⊢ Iff (And True (ContDiffWithinAtProp I I' n (Function.comp (Function.comp (↑( …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem contMDiffAt_iff_target_of_mem_source
    [SmoothManifoldWithCorners I' M'] {x : M} {y : M'} (hy : f x ∈ (chartAt H' y).source) :
    ContMDiffAt I I' n f x ↔
      ContinuousAt f x ∧ ContMDiffAt I 𝓘(𝕜, E') n (extChartAt I' y ∘ f) x := by
  rw [ContMDiffAt, contMDiffWithinAt_iff_target_of_mem_source hy, continuousWithinAt_univ,
    ContMDiffAt]


theorem contMDiffWithinAt_iff_of_mem_maximalAtlas {x : M} (he : e ∈ maximalAtlas I M)
    (he' : e' ∈ maximalAtlas I' M') (hx : x ∈ e.source) (hy : f x ∈ e'.source) :
    ContMDiffWithinAt I I' n f s x ↔
      ContinuousWithinAt f s x ∧
        ContDiffWithinAt 𝕜 n (e'.extend I' ∘ f ∘ (e.extend I).symm)
          ((e.extend I).symm ⁻¹' s ∩ range I) (e.extend I x) :=
  (contDiffWithinAt_localInvariantProp n).liftPropWithinAt_indep_chart he hx he' hy


/-- An alternative formulation of `contMDiffWithinAt_iff_of_mem_maximalAtlas`
  if the set if `s` lies in `e.source`. -/
theorem contMDiffWithinAt_iff_image {x : M} (he : e ∈ maximalAtlas I M)
    (he' : e' ∈ maximalAtlas I' M') (hs : s ⊆ e.source) (hx : x ∈ e.source) (hy : f x ∈ e'.source) :
    ContMDiffWithinAt I I' n f s x ↔
      ContinuousWithinAt f s x ∧
        ContDiffWithinAt 𝕜 n (e'.extend I' ∘ f ∘ (e.extend I).symm) (e.extend I '' s)
          (e.extend I x) := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    e : PartialHomeomorph M H
    e' : PartialHomeomorph M' H'
    f : M → M'
    s : Set M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    he' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I' M') e'
    hs : HasSubset.Subset s e.source
    hx : Membership.mem e.source x
    hy : Membership.mem e'.source (f x)
    ⊢ Iff (ContMDiffWithinAt I I' n f s x) (And (ContinuousWithinAt f s x) (ContDi …
  -/
  rw [contMDiffWithinAt_iff_of_mem_maximalAtlas he he' hx hy, and_congr_right_iff]
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    e : PartialHomeomorph M H
    e' : PartialHomeomorph M' H'
    f : M → M'
    s : Set M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    he' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I' M') e'
    hs : HasSubset.Subset s e.source
    hx : Membership.mem e.source x
    hy : Membership.mem e'.source (f x)
    ⊢ ContinuousWithinAt f s x → Iff (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑(e' …
  -/
  refine fun _ => contDiffWithinAt_congr_set ?_
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    e : PartialHomeomorph M H
    e' : PartialHomeomorph M' H'
    f : M → M'
    s : Set M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    he' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I' M') e'
    hs : HasSubset.Subset s e.source
    hx : Membership.mem e.source x
    hy : Membership.mem e'.source (f x)
    x✝ : ContinuousWithinAt f s x
    ⊢ (nhds (↑(e.extend I) x)).EventuallyEq (Inter.inter (Set.preimage (↑(e.extend …
  -/
  simp_rw [e.extend_symm_preimage_inter_range_eventuallyEq hs hx]
  /-
    🎉 no goals
  -/


/-- One can reformulate smoothness within a set at a point as continuity within this set at this
point, and smoothness in any chart containing that point. -/
theorem contMDiffWithinAt_iff_of_mem_source {x' : M} {y : M'} (hx : x' ∈ (chartAt H x).source)
    (hy : f x' ∈ (chartAt H' y).source) :
    ContMDiffWithinAt I I' n f s x' ↔
      ContinuousWithinAt f s x' ∧
        ContDiffWithinAt 𝕜 n (extChartAt I' y ∘ f ∘ (extChartAt I x).symm)
          ((extChartAt I x).symm ⁻¹' s ∩ range I) (extChartAt I x x') :=
  contMDiffWithinAt_iff_of_mem_maximalAtlas (chart_mem_maximalAtlas x)
    (chart_mem_maximalAtlas y) hx hy


theorem contMDiffWithinAt_iff_of_mem_source' {x' : M} {y : M'} (hx : x' ∈ (chartAt H x).source)
    (hy : f x' ∈ (chartAt H' y).source) :
    ContMDiffWithinAt I I' n f s x' ↔
      ContinuousWithinAt f s x' ∧
        ContDiffWithinAt 𝕜 n (extChartAt I' y ∘ f ∘ (extChartAt I x).symm)
          ((extChartAt I x).target ∩ (extChartAt I x).symm ⁻¹' (s ∩ f ⁻¹' (extChartAt I' y).source))
          (extChartAt I x x') := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x' : M
    y : M'
    hx : Membership.mem (chartAt H x).source x'
    hy : Membership.mem (chartAt H' y).source (f x')
    ⊢ Iff (ContMDiffWithinAt I I' n f s x') (And (ContinuousWithinAt f s x') (Cont …
  -/
  refine (contMDiffWithinAt_iff_of_mem_source hx hy).trans ?_
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x' : M
    y : M'
    hx : Membership.mem (chartAt H x).source x'
    hy : Membership.mem (chartAt H' y).source (f x')
    ⊢ Iff (And (ContinuousWithinAt f s x') (ContDiffWithinAt 𝕜 (↑n) (Function.comp …
  -/
  rw [← extChartAt_source I] at hx
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x' : M
    y : M'
    hx : Membership.mem (extChartAt I x).source x'
    hy : Membership.mem (chartAt H' y).source (f x')
    ⊢ Iff (And (ContinuousWithinAt f s x') (ContDiffWithinAt 𝕜 (↑n) (Function.comp …
  -/
  rw [← extChartAt_source I'] at hy
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x' : M
    y : M'
    hx : Membership.mem (extChartAt I x).source x'
    hy : Membership.mem (extChartAt I' y).source (f x')
    ⊢ Iff (And (ContinuousWithinAt f s x') (ContDiffWithinAt 𝕜 (↑n) (Function.comp …
  -/
  rw [and_congr_right_iff]
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x' : M
    y : M'
    hx : Membership.mem (extChartAt I x).source x'
    hy : Membership.mem (extChartAt I' y).source (f x')
    ⊢ ContinuousWithinAt f s x' → Iff (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑(e …
  -/
  set e := extChartAt I x; set e' := extChartAt I' (f x)
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x' : M
    y : M'
    hy : Membership.mem (extChartAt I' y).source (f x')
    e : PartialEquiv M E := extChartAt I x
    hx : Membership.mem e.source x'
    e' : PartialEquiv M' E' := extChartAt I' (f x)
    ⊢ ContinuousWithinAt f s x' → Iff (ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑(e …
  -/
  refine fun hc => contDiffWithinAt_congr_set ?_
  rw [← nhdsWithin_eq_iff_eventuallyEq, ← e.image_source_inter_eq',
    ← map_extChartAt_nhdsWithin_eq_image' hx,
    ← map_extChartAt_nhdsWithin' hx, inter_comm, nhdsWithin_inter_of_mem]
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x' : M
    y : M'
    hy : Membership.mem (extChartAt I' y).source (f x')
    e : PartialEquiv M E := extChartAt I x
    hx : Membership.mem e.source x'
    e' : PartialEquiv M' E' := extChartAt I' (f x)
    hc : ContinuousWithinAt f s x'
    ⊢ Membership.mem (nhdsWithin x' s) (Set.preimage f (extChartAt I' y).source)
  -/
  exact hc (extChartAt_source_mem_nhds' hy)
  /-
    🎉 no goals
  -/


theorem contMDiffAt_iff_of_mem_source {x' : M} {y : M'} (hx : x' ∈ (chartAt H x).source)
    (hy : f x' ∈ (chartAt H' y).source) :
    ContMDiffAt I I' n f x' ↔
      ContinuousAt f x' ∧
        ContDiffWithinAt 𝕜 n (extChartAt I' y ∘ f ∘ (extChartAt I x).symm) (range I)
          (extChartAt I x x') :=
  (contMDiffWithinAt_iff_of_mem_source hx hy).trans <| by
    /-
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      x : M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      x' : M
      y : M'
      hx : Membership.mem (chartAt H x).source x'
      hy : Membership.mem (chartAt H' y).source (f x')
      ⊢ Iff (And (ContinuousWithinAt f Set.univ x') (ContDiffWithinAt 𝕜 (↑n) (Functi …
    -/
    rw [continuousWithinAt_univ, preimage_univ, univ_inter]
    /-
      🎉 no goals
    -/


theorem contMDiffOn_iff_of_mem_maximalAtlas (he : e ∈ maximalAtlas I M)
    (he' : e' ∈ maximalAtlas I' M') (hs : s ⊆ e.source) (h2s : MapsTo f s e'.source) :
    ContMDiffOn I I' n f s ↔
      ContinuousOn f s ∧
        ContDiffOn 𝕜 n (e'.extend I' ∘ f ∘ (e.extend I).symm) (e.extend I '' s) := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    e : PartialHomeomorph M H
    e' : PartialHomeomorph M' H'
    f : M → M'
    s : Set M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    he' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I' M') e'
    hs : HasSubset.Subset s e.source
    h2s : Set.MapsTo f s e'.source
    ⊢ Iff (ContMDiffOn I I' n f s) (And (ContinuousOn f s) (ContDiffOn 𝕜 (↑n) (Fun …
  -/
  simp_rw [ContinuousOn, ContDiffOn, Set.forall_mem_image, ← forall_and, ContMDiffOn]
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    e : PartialHomeomorph M H
    e' : PartialHomeomorph M' H'
    f : M → M'
    s : Set M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    he' : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I' M') e'
    hs : HasSubset.Subset s e.source
    h2s : Set.MapsTo f s e'.source
    ⊢ Iff (∀ (x : M), Membership.mem s x → ContMDiffWithinAt I I' n f s x) (∀ (x : …
  -/
  exact forall₂_congr fun x hx => contMDiffWithinAt_iff_image he he' hs (hs hx) (h2s hx)
  /-
    🎉 no goals
  -/


theorem contMDiffOn_iff_of_mem_maximalAtlas' (he : e ∈ maximalAtlas I M)
    (he' : e' ∈ maximalAtlas I' M') (hs : s ⊆ e.source) (h2s : MapsTo f s e'.source) :
    ContMDiffOn I I' n f s ↔
      ContDiffOn 𝕜 n (e'.extend I' ∘ f ∘ (e.extend I).symm) (e.extend I '' s) :=
  (contMDiffOn_iff_of_mem_maximalAtlas he he' hs h2s).trans <| and_iff_right_of_imp fun h ↦
    (e.continuousOn_writtenInExtend_iff hs h2s).1 h.continuousOn


/-- If the set where you want `f` to be smooth lies entirely in a single chart, and `f` maps it
  into a single chart, the smoothness of `f` on that set can be expressed by purely looking in
  these charts.
  Note: this lemma uses `extChartAt I x '' s` instead of `(extChartAt I x).symm ⁻¹' s` to ensure
  that this set lies in `(extChartAt I x).target`. -/
theorem contMDiffOn_iff_of_subset_source {x : M} {y : M'} (hs : s ⊆ (chartAt H x).source)
    (h2s : MapsTo f s (chartAt H' y).source) :
    ContMDiffOn I I' n f s ↔
      ContinuousOn f s ∧
        ContDiffOn 𝕜 n (extChartAt I' y ∘ f ∘ (extChartAt I x).symm) (extChartAt I x '' s) :=
  contMDiffOn_iff_of_mem_maximalAtlas (chart_mem_maximalAtlas x) (chart_mem_maximalAtlas y) hs
    h2s


/-- If the set where you want `f` to be smooth lies entirely in a single chart, and `f` maps it
  into a single chart, the smoothness of `f` on that set can be expressed by purely looking in
  these charts.
  Note: this lemma uses `extChartAt I x '' s` instead of `(extChartAt I x).symm ⁻¹' s` to ensure
  that this set lies in `(extChartAt I x).target`. -/
theorem contMDiffOn_iff_of_subset_source' {x : M} {y : M'} (hs : s ⊆ (extChartAt I x).source)
    (h2s : MapsTo f s (extChartAt I' y).source) :
    ContMDiffOn I I' n f s ↔
        ContDiffOn 𝕜 n (extChartAt I' y ∘ f ∘ (extChartAt I x).symm) (extChartAt I x '' s) := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    x : M
    y : M'
    hs : HasSubset.Subset s (extChartAt I x).source
    h2s : Set.MapsTo f s (extChartAt I' y).source
    ⊢ Iff (ContMDiffOn I I' n f s) (ContDiffOn 𝕜 (↑n) (Function.comp (↑(extChartAt …
  -/
  rw [extChartAt_source] at hs h2s
  exact contMDiffOn_iff_of_mem_maximalAtlas' (chart_mem_maximalAtlas x)
    (chart_mem_maximalAtlas y) hs h2s


/-- One can reformulate smoothness on a set as continuity on this set, and smoothness in any
extended chart. -/
theorem contMDiffOn_iff :
    ContMDiffOn I I' n f s ↔
      ContinuousOn f s ∧
        ∀ (x : M) (y : M'),
          ContDiffOn 𝕜 n (extChartAt I' y ∘ f ∘ (extChartAt I x).symm)
            ((extChartAt I x).target ∩
              (extChartAt I x).symm ⁻¹' (s ∩ f ⁻¹' (extChartAt I' y).source)) := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    ⊢ Iff (ContMDiffOn I I' n f s) (And (ContinuousOn f s) (∀ (x : M) (y : M'), Co …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      ⊢ ContMDiffOn I I' n f s → And (ContinuousOn f s) (∀ (x : M) (y : M'), ContDif …
    -/
  · intro h
    /-
      case mp
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContMDiffOn I I' n f s
      ⊢ And (ContinuousOn f s) (∀ (x : M) (y : M'), ContDiffOn 𝕜 (↑n) (Function.comp …
    -/
    refine ⟨fun x hx => (h x hx).1, fun x y z hz => ?_⟩
    /-
      case mp
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContMDiffOn I I' n f s
      x : M
      y : M'
      z : E
      hz : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(extC …
      ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑(extChartAt I' y)) (Function.comp f …
    -/
    simp only [mfld_simps] at hz
    /-
      case mp
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContMDiffOn I I' n f s
      x : M
      y : M'
      z : E
      hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
      ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑(extChartAt I' y)) (Function.comp f …
    -/
    let w := (extChartAt I x).symm z
    /-
      case mp
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContMDiffOn I I' n f s
      x : M
      y : M'
      z : E
      hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
      w : M := ↑(extChartAt I x).symm z
      ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑(extChartAt I' y)) (Function.comp f …
    -/
    have : w ∈ s := by simp only [w, hz, mfld_simps]
    /-
      case mp
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContMDiffOn I I' n f s
      x : M
      y : M'
      z : E
      hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
      w : M := ↑(extChartAt I x).symm z
      this : Membership.mem s w
      ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑(extChartAt I' y)) (Function.comp f …
    -/
    specialize h w this
    /-
      case mp
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      x : M
      y : M'
      z : E
      hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
      w : M := ↑(extChartAt I x).symm z
      this : Membership.mem s w
      h : ContMDiffWithinAt I I' n f s w
      ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑(extChartAt I' y)) (Function.comp f …
    -/
    have w1 : w ∈ (chartAt H x).source := by simp only [w, hz, mfld_simps]
    /-
      case mp
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      x : M
      y : M'
      z : E
      hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
      w : M := ↑(extChartAt I x).symm z
      this : Membership.mem s w
      h : ContMDiffWithinAt I I' n f s w
      w1 : Membership.mem (chartAt H x).source w
      ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑(extChartAt I' y)) (Function.comp f …
    -/
    have w2 : f w ∈ (chartAt H' y).source := by simp only [w, hz, mfld_simps]
    /-
      case mp
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      x : M
      y : M'
      z : E
      hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
      w : M := ↑(extChartAt I x).symm z
      this : Membership.mem s w
      h : ContMDiffWithinAt I I' n f s w
      w1 : Membership.mem (chartAt H x).source w
      w2 : Membership.mem (chartAt H' y).source (f w)
      ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑(extChartAt I' y)) (Function.comp f …
    -/
    convert ((contMDiffWithinAt_iff_of_mem_source w1 w2).mp h).2.mono _
      /-
        case h.e'_12
        𝕜 : Type u_1
        inst✝¹² : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹¹ : NormedAddCommGroup E
        inst✝¹⁰ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝⁸ : TopologicalSpace M
        inst✝⁷ : ChartedSpace H M
        E' : Type u_5
        inst✝⁶ : NormedAddCommGroup E'
        inst✝⁵ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝⁴ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M' : Type u_7
        inst✝³ : TopologicalSpace M'
        inst✝² : ChartedSpace H' M'
        f : M → M'
        s : Set M
        n : ENat
        inst✝¹ : SmoothManifoldWithCorners I M
        inst✝ : SmoothManifoldWithCorners I' M'
        x : M
        y : M'
        z : E
        hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
        w : M := ↑(extChartAt I x).symm z
        this : Membership.mem s w
        h : ContMDiffWithinAt I I' n f s w
        w1 : Membership.mem (chartAt H x).source w
        w2 : Membership.mem (chartAt H' y).source (f w)
        ⊢ Eq z (↑(extChartAt I x) w)
      -/
    · simp only [w, hz, mfld_simps]
      /-
        🎉 no goals
      -/
      /-
        case mp.convert_2
        𝕜 : Type u_1
        inst✝¹² : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹¹ : NormedAddCommGroup E
        inst✝¹⁰ : NormedSpace 𝕜 E
        H : Type u_3
        inst✝⁹ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        M : Type u_4
        inst✝⁸ : TopologicalSpace M
        inst✝⁷ : ChartedSpace H M
        E' : Type u_5
        inst✝⁶ : NormedAddCommGroup E'
        inst✝⁵ : NormedSpace 𝕜 E'
        H' : Type u_6
        inst✝⁴ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E' H'
        M' : Type u_7
        inst✝³ : TopologicalSpace M'
        inst✝² : ChartedSpace H' M'
        f : M → M'
        s : Set M
        n : ENat
        inst✝¹ : SmoothManifoldWithCorners I M
        inst✝ : SmoothManifoldWithCorners I' M'
        x : M
        y : M'
        z : E
        hz : And (And (Membership.mem (Set.range ↑I) z) (Membership.mem (chartAt H x). …
        w : M := ↑(extChartAt I x).symm z
        this : Membership.mem s w
        h : ContMDiffWithinAt I I' n f s w
        w1 : Membership.mem (chartAt H x).source w
        w2 : Membership.mem (chartAt H' y).source (f w)
        ⊢ HasSubset.Subset (Inter.inter (extChartAt I x).target (Set.preimage (↑(extCh …
      -/
    · mfld_set_tac
      /-
        🎉 no goals
      -/
    /-
      case mpr
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      ⊢ And (ContinuousOn f s) (∀ (x : M) (y : M'), ContDiffOn 𝕜 (↑n) (Function.comp …
    -/
  · rintro ⟨hcont, hdiff⟩ x hx
    /-
      case mpr.intro
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      hcont : ContinuousOn f s
      hdiff : ∀ (x : M) (y : M'), ContDiffOn 𝕜 (↑n) (Function.comp (↑(extChartAt I'  …
      x : M
      hx : Membership.mem s x
      ⊢ ContMDiffWithinAt I I' n f s x
    -/
    refine (contDiffWithinAt_localInvariantProp n).liftPropWithinAt_iff.mpr ?_
    /-
      case mpr.intro
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      hcont : ContinuousOn f s
      hdiff : ∀ (x : M) (y : M'), ContDiffOn 𝕜 (↑n) (Function.comp (↑(extChartAt I'  …
      x : M
      hx : Membership.mem s x
      ⊢ And (ContinuousWithinAt f s x) (ContDiffWithinAtProp I I' n (Function.comp ( …
    -/
    refine ⟨hcont x hx, ?_⟩
    /-
      case mpr.intro
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      hcont : ContinuousOn f s
      hdiff : ∀ (x : M) (y : M'), ContDiffOn 𝕜 (↑n) (Function.comp (↑(extChartAt I'  …
      x : M
      hx : Membership.mem s x
      ⊢ ContDiffWithinAtProp I I' n (Function.comp (↑(chartAt H' (f x))) (Function.c …
    -/
    dsimp [ContDiffWithinAtProp]
    /-
      case mpr.intro
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      hcont : ContinuousOn f s
      hdiff : ∀ (x : M) (y : M'), ContDiffOn 𝕜 (↑n) (Function.comp (↑(extChartAt I'  …
      x : M
      hx : Membership.mem s x
      ⊢ ContDiffWithinAt 𝕜 (↑n) (Function.comp (↑I') (Function.comp (Function.comp ( …
    -/
    convert hdiff x (f x) (extChartAt I x x) (by simp only [hx, mfld_simps]) using 1
    /-
      case h.e'_11
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      hcont : ContinuousOn f s
      hdiff : ∀ (x : M) (y : M'), ContDiffOn 𝕜 (↑n) (Function.comp (↑(extChartAt I'  …
      x : M
      hx : Membership.mem s x
      ⊢ Eq (Inter.inter (Inter.inter (Set.preimage (↑I.symm) (chartAt H x).target) ( …
    -/
    mfld_set_tac
    /-
      🎉 no goals
    -/


/-- zero-smoothness on a set is equivalent to continuity on this set. -/
theorem contMDiffOn_zero_iff :
    ContMDiffOn I I' 0 f s ↔ ContinuousOn f s := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    ⊢ Iff (ContMDiffOn I I' 0 f s) (ContinuousOn f s)
  -/
  rw [contMDiffOn_iff]
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    ⊢ Iff (And (ContinuousOn f s) (∀ (x : M) (y : M'), ContDiffOn 𝕜 (↑0) (Function …
  -/
  refine ⟨fun h ↦ h.1, fun h ↦ ⟨h, ?_⟩⟩
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    h : ContinuousOn f s
    ⊢ ∀ (x : M) (y : M'), ContDiffOn 𝕜 (↑0) (Function.comp (↑(extChartAt I' y)) (F …
  -/
  intro x y
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    h : ContinuousOn f s
    x : M
    y : M'
    ⊢ ContDiffOn 𝕜 (↑0) (Function.comp (↑(extChartAt I' y)) (Function.comp f ↑(ext …
  -/
  rw [WithTop.coe_zero, contDiffOn_zero]
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    h : ContinuousOn f s
    x : M
    y : M'
    ⊢ ContinuousOn (Function.comp (↑(extChartAt I' y)) (Function.comp f ↑(extChart …
  -/
  apply (continuousOn_extChartAt _).comp
    /-
      case hf
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      x : M
      y : M'
      ⊢ ContinuousOn (Function.comp f ↑(extChartAt I x).symm) (Inter.inter (extChart …
    -/
  · apply h.comp ((continuousOn_extChartAt_symm _).mono inter_subset_left) (fun z hz ↦ ?_)
    /-
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      x : M
      y : M'
      z : E
      hz : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(extC …
      ⊢ Membership.mem s (↑(extChartAt I x).symm z)
    -/
    simp only [preimage_inter, mem_inter_iff, mem_preimage] at hz
    /-
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      x : M
      y : M'
      z : E
      hz : And (Membership.mem (extChartAt I x).target z) (And (Membership.mem s (↑( …
      ⊢ Membership.mem s (↑(extChartAt I x).symm z)
    -/
    exact hz.2.1
    /-
      🎉 no goals
    -/
    /-
      case h
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      x : M
      y : M'
      ⊢ Set.MapsTo (Function.comp f ↑(extChartAt I x).symm) (Inter.inter (extChartAt …
    -/
  · intro z hz
    /-
      case h
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      x : M
      y : M'
      z : E
      hz : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(extC …
      ⊢ Membership.mem (extChartAt I' y).source (Function.comp f (↑(extChartAt I x). …
    -/
    simp only [preimage_inter, mem_inter_iff, mem_preimage] at hz
    /-
      case h
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      x : M
      y : M'
      z : E
      hz : And (Membership.mem (extChartAt I x).target z) (And (Membership.mem s (↑( …
      ⊢ Membership.mem (extChartAt I' y).source (Function.comp f (↑(extChartAt I x). …
    -/
    exact hz.2.2
    /-
      🎉 no goals
    -/


/-- One can reformulate smoothness on a set as continuity on this set, and smoothness in any
extended chart in the target. -/
theorem contMDiffOn_iff_target :
    ContMDiffOn I I' n f s ↔
      ContinuousOn f s ∧
        ∀ y : M',
          ContMDiffOn I 𝓘(𝕜, E') n (extChartAt I' y ∘ f) (s ∩ f ⁻¹' (extChartAt I' y).source) := by
  simp only [contMDiffOn_iff, ModelWithCorners.source_eq, chartAt_self_eq,
    PartialHomeomorph.refl_partialEquiv, PartialEquiv.refl_trans, extChartAt,
    PartialHomeomorph.extend, Set.preimage_univ, Set.inter_univ, and_congr_right_iff]
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    ⊢ ContinuousOn f s → Iff (∀ (x : M) (y : M'), ContDiffOn 𝕜 (↑n) (Function.comp …
  -/
  intro h
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    h : ContinuousOn f s
    ⊢ Iff (∀ (x : M) (y : M'), ContDiffOn 𝕜 (↑n) (Function.comp (↑((chartAt H' y). …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      ⊢ (∀ (x : M) (y : M'), ContDiffOn 𝕜 (↑n) (Function.comp (↑((chartAt H' y).tran …
    -/
  · refine fun h' y => ⟨?_, fun x _ => h' x y⟩
    /-
      case mp
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      h' : ∀ (x : M) (y : M'), ContDiffOn 𝕜 (↑n) (Function.comp (↑((chartAt H' y).tr …
      y : M'
      ⊢ ContinuousOn (Function.comp (↑((chartAt H' y).trans I'.toPartialEquiv)) f) ( …
    -/
    have h'' : ContinuousOn _ univ := (ModelWithCorners.continuous I').continuousOn
    /-
      case mp
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      h' : ∀ (x : M) (y : M'), ContDiffOn 𝕜 (↑n) (Function.comp (↑((chartAt H' y).tr …
      y : M'
      h'' : ContinuousOn (↑I') Set.univ
      ⊢ ContinuousOn (Function.comp (↑((chartAt H' y).trans I'.toPartialEquiv)) f) ( …
    -/
    convert (h''.comp_inter (chartAt H' y).continuousOn_toFun).comp_inter h
    /-
      case h.e'_6.h.e'_4.h.e'_4
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      h' : ∀ (x : M) (y : M'), ContDiffOn 𝕜 (↑n) (Function.comp (↑((chartAt H' y).tr …
      y : M'
      h'' : ContinuousOn (↑I') Set.univ
      ⊢ Eq ((chartAt H' y).trans I'.toPartialEquiv).source (Inter.inter (chartAt H'  …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      n : ENat
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      h : ContinuousOn f s
      ⊢ (∀ (y : M'), And (ContinuousOn (Function.comp (↑((chartAt H' y).trans I'.toP …
    -/
  · exact fun h' x y => (h' y).2 x 0
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-20")] alias smoothOn_iff := contMDiffOn_iff


@[deprecated (since := "2024-11-20")] alias smoothOn_iff_target := contMDiffOn_iff_target



/-- One can reformulate smoothness as continuity and smoothness in any extended chart. -/
theorem contMDiff_iff :
    ContMDiff I I' n f ↔
      Continuous f ∧
        ∀ (x : M) (y : M'),
          ContDiffOn 𝕜 n (extChartAt I' y ∘ f ∘ (extChartAt I x).symm)
            ((extChartAt I x).target ∩
              (extChartAt I x).symm ⁻¹' (f ⁻¹' (extChartAt I' y).source)) := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    ⊢ Iff (ContMDiff I I' n f) (And (Continuous f) (∀ (x : M) (y : M'), ContDiffOn …
  -/
  simp [← contMDiffOn_univ, contMDiffOn_iff, continuous_iff_continuousOn_univ]
  /-
    🎉 no goals
  -/


/-- One can reformulate smoothness as continuity and smoothness in any extended chart in the
target. -/
theorem contMDiff_iff_target :
    ContMDiff I I' n f ↔
      Continuous f ∧ ∀ y : M',
        ContMDiffOn I 𝓘(𝕜, E') n (extChartAt I' y ∘ f) (f ⁻¹' (extChartAt I' y).source) := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    ⊢ Iff (ContMDiff I I' n f) (And (Continuous f) (∀ (y : M'), ContMDiffOn I (mod …
  -/
  rw [← contMDiffOn_univ, contMDiffOn_iff_target]
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    ⊢ Iff (And (ContinuousOn f Set.univ) (∀ (y : M'), ContMDiffOn I (modelWithCorn …
  -/
  simp [continuous_iff_continuousOn_univ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-20")] alias smooth_iff := contMDiff_iff


@[deprecated (since := "2024-11-20")] alias smooth_iff_target := contMDiff_iff_target


/-- zero-smoothness is equivalent to continuity. -/
theorem contMDiff_zero_iff :
    ContMDiff I I' 0 f ↔ Continuous f := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    ⊢ Iff (ContMDiff I I' 0 f) (Continuous f)
  -/
  rw [← contMDiffOn_univ, continuous_iff_continuousOn_univ, contMDiffOn_zero_iff]
  /-
    🎉 no goals
  -/


theorem ContMDiffWithinAt.of_succ {n : ℕ} (h : ContMDiffWithinAt I I' n.succ f s x) :
    ContMDiffWithinAt I I' n f s x :=
  h.of_le (WithTop.coe_le_coe.2 (Nat.le_succ n))


theorem ContMDiffAt.of_succ {n : ℕ} (h : ContMDiffAt I I' n.succ f x) : ContMDiffAt I I' n f x :=
  ContMDiffWithinAt.of_succ h


theorem ContMDiffOn.of_succ {n : ℕ} (h : ContMDiffOn I I' n.succ f s) : ContMDiffOn I I' n f s :=
  fun x hx => (h x hx).of_succ


theorem ContMDiff.of_succ {n : ℕ} (h : ContMDiff I I' n.succ f) : ContMDiff I I' n f := fun x =>
  (h x).of_succ


theorem ContMDiffWithinAt.continuousWithinAt (hf : ContMDiffWithinAt I I' n f s x) :
    ContinuousWithinAt f s x :=
  hf.1


theorem ContMDiffAt.continuousAt (hf : ContMDiffAt I I' n f x) : ContinuousAt f x :=
  (continuousWithinAt_univ _ _).1 <| ContMDiffWithinAt.continuousWithinAt hf


theorem ContMDiffOn.continuousOn (hf : ContMDiffOn I I' n f s) : ContinuousOn f s := fun x hx =>
  (hf x hx).continuousWithinAt


theorem ContMDiff.continuous (hf : ContMDiff I I' n f) : Continuous f :=
  continuous_iff_continuousAt.2 fun x => (hf x).continuousAt


theorem contMDiffWithinAt_top :
    ContMDiffWithinAt I I' ⊤ f s x ↔ ∀ n : ℕ, ContMDiffWithinAt I I' n f s x :=
  ⟨fun h n => ⟨h.1, contDiffWithinAt_infty.1 h.2 n⟩, fun H =>
    ⟨(H 0).1, contDiffWithinAt_infty.2 fun n => (H n).2⟩⟩


theorem contMDiffAt_top : ContMDiffAt I I' ⊤ f x ↔ ∀ n : ℕ, ContMDiffAt I I' n f x :=
  contMDiffWithinAt_top


theorem contMDiffOn_top : ContMDiffOn I I' ⊤ f s ↔ ∀ n : ℕ, ContMDiffOn I I' n f s :=
  ⟨fun h _ => h.of_le le_top, fun h x hx => contMDiffWithinAt_top.2 fun n => h n x hx⟩


theorem contMDiff_top : ContMDiff I I' ⊤ f ↔ ∀ n : ℕ, ContMDiff I I' n f :=
  ⟨fun h _ => h.of_le le_top, fun h x => contMDiffWithinAt_top.2 fun n => h n x⟩


theorem contMDiffWithinAt_iff_nat :
    ContMDiffWithinAt I I' n f s x ↔ ∀ m : ℕ, (m : ℕ∞) ≤ n → ContMDiffWithinAt I I' m f s x := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    ⊢ Iff (ContMDiffWithinAt I I' n f s x) (∀ (m : Nat), LE.le (↑m) n → ContMDiffW …
  -/
  refine ⟨fun h m hm => h.of_le hm, fun h => ?_⟩
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    h : ∀ (m : Nat), LE.le (↑m) n → ContMDiffWithinAt I I' (↑m) f s x
    ⊢ ContMDiffWithinAt I I' n f s x
  -/
  cases' n with n
    /-
      case top
      𝕜 : Type u_1
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : ChartedSpace H M
      E' : Type u_5
      inst✝⁴ : NormedAddCommGroup E'
      inst✝³ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝² : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      f : M → M'
      s : Set M
      x : M
      h : ∀ (m : Nat), LE.le (↑m) Top.top → ContMDiffWithinAt I I' (↑m) f s x
      ⊢ ContMDiffWithinAt I I' Top.top f s x
    -/
  · exact contMDiffWithinAt_top.2 fun n => h n le_top
    /-
      🎉 no goals
    -/
    /-
      case coe
      𝕜 : Type u_1
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : ChartedSpace H M
      E' : Type u_5
      inst✝⁴ : NormedAddCommGroup E'
      inst✝³ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝² : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝¹ : TopologicalSpace M'
      inst✝ : ChartedSpace H' M'
      f : M → M'
      s : Set M
      x : M
      n : Nat
      h : ∀ (m : Nat), LE.le ↑m ↑n → ContMDiffWithinAt I I' (↑m) f s x
      ⊢ ContMDiffWithinAt I I' (↑n) f s x
    -/
  · exact h n le_rfl
    /-
      🎉 no goals
    -/


theorem ContMDiffWithinAt.mono_of_mem_nhdsWithin
    (hf : ContMDiffWithinAt I I' n f s x) (hts : s ∈ 𝓝[t] x) :
    ContMDiffWithinAt I I' n f t x :=
  StructureGroupoid.LocalInvariantProp.liftPropWithinAt_mono_of_mem_nhdsWithin
    (contDiffWithinAtProp_mono_of_mem_nhdsWithin n) hf hts


@[deprecated (since := "2024-10-31")]
alias ContMDiffWithinAt.mono_of_mem := ContMDiffWithinAt.mono_of_mem_nhdsWithin


theorem ContMDiffWithinAt.mono (hf : ContMDiffWithinAt I I' n f s x) (hts : t ⊆ s) :
    ContMDiffWithinAt I I' n f t x :=
  hf.mono_of_mem_nhdsWithin <| mem_of_superset self_mem_nhdsWithin hts


theorem contMDiffWithinAt_congr_set (h : s =ᶠ[𝓝 x] t) :
    ContMDiffWithinAt I I' n f s x ↔ ContMDiffWithinAt I I' n f t x :=
  (contDiffWithinAt_localInvariantProp n).liftPropWithinAt_congr_set h


theorem ContMDiffWithinAt.congr_set (h : ContMDiffWithinAt I I' n f s x) (hst : s =ᶠ[𝓝 x] t) :
    ContMDiffWithinAt I I' n f t x :=
  (contMDiffWithinAt_congr_set hst).1 h


@[deprecated (since := "2024-10-23")]
alias contMDiffWithinAt_congr_nhds := contMDiffWithinAt_congr_set


theorem contMDiffWithinAt_insert_self :
    ContMDiffWithinAt I I' n f (insert x s) x ↔ ContMDiffWithinAt I I' n f s x := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    ⊢ Iff (ContMDiffWithinAt I I' n f (Insert.insert x s) x) (ContMDiffWithinAt I  …
  -/
  simp only [contMDiffWithinAt_iff, continuousWithinAt_insert_self]
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    ⊢ Iff (And (ContinuousWithinAt f s x) (ContDiffWithinAt 𝕜 (↑n) (Function.comp  …
  -/
  refine Iff.rfl.and <| (contDiffWithinAt_congr_set ?_).trans contDiffWithinAt_insert_self
  simp only [← map_extChartAt_nhdsWithin, nhdsWithin_insert, Filter.map_sup, Filter.map_pure,
    ← nhdsWithin_eq_iff_eventuallyEq]


alias ⟨ContMDiffWithinAt.of_insert, _⟩ := contMDiffWithinAt_insert_self

-- TODO: use `alias` again once it can make protected theorems

protected theorem ContMDiffWithinAt.insert (h : ContMDiffWithinAt I I' n f s x) :
    ContMDiffWithinAt I I' n f (insert x s) x :=
  contMDiffWithinAt_insert_self.2 h


/-- Being `C^n` in a set only depends on the germ of the set. Version where one only requires
the two sets to coincide locally in the complement of a point `y`. -/
theorem contMDiffWithinAt_congr_set' (y : M) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) :
    ContMDiffWithinAt I I' n f s x ↔ ContMDiffWithinAt I I' n f t x := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    f : M → M'
    s t : Set M
    x : M
    n : ENat
    y : M
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    ⊢ Iff (ContMDiffWithinAt I I' n f s x) (ContMDiffWithinAt I I' n f t x)
  -/
  have : T1Space M := I.t1Space M
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    f : M → M'
    s t : Set M
    x : M
    n : ENat
    y : M
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    this : T1Space M
    ⊢ Iff (ContMDiffWithinAt I I' n f s x) (ContMDiffWithinAt I I' n f t x)
  -/
  rw [← contMDiffWithinAt_insert_self (s := s), ← contMDiffWithinAt_insert_self (s := t)]
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝¹ : TopologicalSpace M'
    inst✝ : ChartedSpace H' M'
    f : M → M'
    s t : Set M
    x : M
    n : ENat
    y : M
    h : (nhdsWithin x (HasCompl.compl (Singleton.singleton y))).EventuallyEq s t
    this : T1Space M
    ⊢ Iff (ContMDiffWithinAt I I' n f (Insert.insert x s) x) (ContMDiffWithinAt I  …
  -/
  exact contMDiffWithinAt_congr_set (eventuallyEq_insert h)
  /-
    🎉 no goals
  -/


protected theorem ContMDiffAt.contMDiffWithinAt (hf : ContMDiffAt I I' n f x) :
    ContMDiffWithinAt I I' n f s x :=
  ContMDiffWithinAt.mono hf (subset_univ _)


@[deprecated (since := "2024-11-20")] alias SmoothAt.smoothWithinAt := ContMDiffAt.contMDiffWithinAt


theorem ContMDiffOn.mono (hf : ContMDiffOn I I' n f s) (hts : t ⊆ s) : ContMDiffOn I I' n f t :=
  fun x hx => (hf x (hts hx)).mono hts


protected theorem ContMDiff.contMDiffOn (hf : ContMDiff I I' n f) : ContMDiffOn I I' n f s :=
  fun x _ => (hf x).contMDiffWithinAt


@[deprecated (since := "2024-11-20")] alias Smooth.smoothOn := ContMDiff.contMDiffOn


theorem contMDiffWithinAt_inter' (ht : t ∈ 𝓝[s] x) :
    ContMDiffWithinAt I I' n f (s ∩ t) x ↔ ContMDiffWithinAt I I' n f s x :=
  (contDiffWithinAt_localInvariantProp n).liftPropWithinAt_inter' ht


theorem contMDiffWithinAt_inter (ht : t ∈ 𝓝 x) :
    ContMDiffWithinAt I I' n f (s ∩ t) x ↔ ContMDiffWithinAt I I' n f s x :=
  (contDiffWithinAt_localInvariantProp n).liftPropWithinAt_inter ht


protected theorem ContMDiffWithinAt.contMDiffAt
    (h : ContMDiffWithinAt I I' n f s x) (ht : s ∈ 𝓝 x) :
    ContMDiffAt I I' n f x :=
  (contDiffWithinAt_localInvariantProp n).liftPropAt_of_liftPropWithinAt h ht


@[deprecated (since := "2024-11-20")] alias SmoothWithinAt.smoothAt := ContMDiffWithinAt.contMDiffAt


protected theorem ContMDiffOn.contMDiffAt (h : ContMDiffOn I I' n f s) (hx : s ∈ 𝓝 x) :
    ContMDiffAt I I' n f x :=
  (h x (mem_of_mem_nhds hx)).contMDiffAt hx


@[deprecated (since := "2024-11-20")] alias SmoothOn.smoothAt := ContMDiffOn.contMDiffAt


theorem contMDiffOn_iff_source_of_mem_maximalAtlas [SmoothManifoldWithCorners I M]
    (he : e ∈ maximalAtlas I M) (hs : s ⊆ e.source) :
    ContMDiffOn I I' n f s ↔
      ContMDiffOn 𝓘(𝕜, E) I' n (f ∘ (e.extend I).symm) (e.extend I '' s) := by
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    e : PartialHomeomorph M H
    f : M → M'
    s : Set M
    n : ENat
    inst✝ : SmoothManifoldWithCorners I M
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    hs : HasSubset.Subset s e.source
    ⊢ Iff (ContMDiffOn I I' n f s) (ContMDiffOn (modelWithCornersSelf 𝕜 E) I' n (F …
  -/
  simp_rw [ContMDiffOn, Set.forall_mem_image]
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    e : PartialHomeomorph M H
    f : M → M'
    s : Set M
    n : ENat
    inst✝ : SmoothManifoldWithCorners I M
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    hs : HasSubset.Subset s e.source
    ⊢ Iff (∀ (x : M), Membership.mem s x → ContMDiffWithinAt I I' n f s x) (∀ ⦃x : …
  -/
  refine forall₂_congr fun x hx => ?_
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    e : PartialHomeomorph M H
    f : M → M'
    s : Set M
    n : ENat
    inst✝ : SmoothManifoldWithCorners I M
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    hs : HasSubset.Subset s e.source
    x : M
    hx : Membership.mem s x
    ⊢ Iff (ContMDiffWithinAt I I' n f s x) (ContMDiffWithinAt (modelWithCornersSel …
  -/
  rw [contMDiffWithinAt_iff_source_of_mem_maximalAtlas he (hs hx)]
  /-
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    e : PartialHomeomorph M H
    f : M → M'
    s : Set M
    n : ENat
    inst✝ : SmoothManifoldWithCorners I M
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    hs : HasSubset.Subset s e.source
    x : M
    hx : Membership.mem s x
    ⊢ Iff (ContMDiffWithinAt (modelWithCornersSelf 𝕜 E) I' n (Function.comp f ↑(e. …
  -/
  apply contMDiffWithinAt_congr_set
  /-
    case h
    𝕜 : Type u_1
    inst✝¹¹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁸ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : ChartedSpace H M
    E' : Type u_5
    inst✝⁵ : NormedAddCommGroup E'
    inst✝⁴ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝³ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝² : TopologicalSpace M'
    inst✝¹ : ChartedSpace H' M'
    e : PartialHomeomorph M H
    f : M → M'
    s : Set M
    n : ENat
    inst✝ : SmoothManifoldWithCorners I M
    he : Membership.mem (SmoothManifoldWithCorners.maximalAtlas I M) e
    hs : HasSubset.Subset s e.source
    x : M
    hx : Membership.mem s x
    ⊢ (nhds (↑(e.extend I) x)).EventuallyEq (Inter.inter (Set.preimage (↑(e.extend …
  -/
  simp_rw [e.extend_symm_preimage_inter_range_eventuallyEq hs (hs hx)]
  /-
    🎉 no goals
  -/

-- Porting note: didn't compile; fixed by golfing the proof and moving parts to lemmas

/-- A function is `C^n` within a set at a point, for `n : ℕ`, if and only if it is `C^n` on
a neighborhood of this point. -/
theorem contMDiffWithinAt_iff_contMDiffOn_nhds
    [SmoothManifoldWithCorners I M] [SmoothManifoldWithCorners I' M'] {n : ℕ} :
    ContMDiffWithinAt I I' n f s x ↔ ∃ u ∈ 𝓝[insert x s] x, ContMDiffOn I I' n f u := by
  -- WLOG, `x ∈ s`, otherwise we add `x` to `s`
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    ⊢ Iff (ContMDiffWithinAt I I' (↑n) f s x) (Exists fun u => And (Membership.mem …
  -/
  wlog hxs : x ∈ s generalizing s
    /-
      case inr
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s : Set M
      x : M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      n : Nat
      this : ∀ {s : Set M}, Membership.mem s x → Iff (ContMDiffWithinAt I I' (↑n) f  …
      hxs : Not (Membership.mem s x)
      ⊢ Iff (ContMDiffWithinAt I I' (↑n) f s x) (Exists fun u => And (Membership.mem …
    -/
  · rw [← contMDiffWithinAt_insert_self, this (mem_insert _ _), insert_idem]
    /-
      🎉 no goals
    -/
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s✝ : Set M
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    s : Set M
    hxs : Membership.mem s x
    ⊢ Iff (ContMDiffWithinAt I I' (↑n) f s x) (Exists fun u => And (Membership.mem …
  -/
  rw [insert_eq_of_mem hxs]
  -- The `←` implication is trivial
  refine ⟨fun h ↦ ?_, fun ⟨u, hmem, hu⟩ ↦
    (hu _ (mem_of_mem_nhdsWithin hxs hmem)).mono_of_mem_nhdsWithin hmem⟩
  -- The property is true in charts. Let `v` be a good neighborhood in the chart where the function
  -- is smooth.
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s✝ : Set M
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    s : Set M
    hxs : Membership.mem s x
    h : ContMDiffWithinAt I I' (↑n) f s x
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x s) u) (ContMDiffOn I I' (↑ …
  -/
  rcases (contMDiffWithinAt_iff'.1 h).2.contDiffOn le_rfl (by simp) with ⟨v, hmem, hsub, hv⟩
  have hxs' : extChartAt I x x ∈ (extChartAt I x).target ∩
      (extChartAt I x).symm ⁻¹' (s ∩ f ⁻¹' (extChartAt I' (f x)).source) :=
    ⟨(extChartAt I x).map_source (mem_extChartAt_source _), by rwa [extChartAt_to_inv], by
      rw [extChartAt_to_inv]; apply mem_extChartAt_source⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s✝ : Set M
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    s : Set M
    hxs : Membership.mem s x
    h : ContMDiffWithinAt I I' (↑n) f s x
    v : Set E
    hmem : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Insert.insert (↑(extC …
    hsub : HasSubset.Subset v (Insert.insert (↑(extChartAt I x) x) (Inter.inter (e …
    hv : ContDiffOn 𝕜 (↑↑n) (Function.comp (↑(extChartAt I' (f x))) (Function.comp …
    hxs' : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(ex …
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x s) u) (ContMDiffOn I I' (↑ …
  -/
  rw [insert_eq_of_mem hxs'] at hmem hsub
  -- Then `(extChartAt I x).symm '' v` is the neighborhood we are looking for.
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s✝ : Set M
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    s : Set M
    hxs : Membership.mem s x
    h : ContMDiffWithinAt I I' (↑n) f s x
    v : Set E
    hmem : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (extChart …
    hsub : HasSubset.Subset v (Inter.inter (extChartAt I x).target (Set.preimage ( …
    hv : ContDiffOn 𝕜 (↑↑n) (Function.comp (↑(extChartAt I' (f x))) (Function.comp …
    hxs' : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(ex …
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x s) u) (ContMDiffOn I I' (↑ …
  -/
  refine ⟨(extChartAt I x).symm '' v, ?_, ?_⟩
  · rw [← map_extChartAt_symm_nhdsWithin (I := I),
      h.1.nhdsWithin_extChartAt_symm_preimage_inter_range (I := I) (I' := I')]
    /-
      case intro.intro.intro.refine_1
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s✝ : Set M
      x : M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      n : Nat
      s : Set M
      hxs : Membership.mem s x
      h : ContMDiffWithinAt I I' (↑n) f s x
      v : Set E
      hmem : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (extChart …
      hsub : HasSubset.Subset v (Inter.inter (extChartAt I x).target (Set.preimage ( …
      hv : ContDiffOn 𝕜 (↑↑n) (Function.comp (↑(extChartAt I' (f x))) (Function.comp …
      hxs' : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(ex …
      ⊢ Membership.mem (Filter.map (↑(extChartAt I x).symm) (nhdsWithin (↑(extChartA …
    -/
    exact image_mem_map hmem
    /-
      🎉 no goals
    -/
  · have hv₁ : (extChartAt I x).symm '' v ⊆ (extChartAt I x).source :=
      image_subset_iff.2 fun y hy ↦ (extChartAt I x).map_target (hsub hy).1
    have hv₂ : MapsTo f ((extChartAt I x).symm '' v) (extChartAt I' (f x)).source := by
      rintro _ ⟨y, hy, rfl⟩
      exact (hsub hy).2.2
    /-
      case intro.intro.intro.refine_2
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s✝ : Set M
      x : M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      n : Nat
      s : Set M
      hxs : Membership.mem s x
      h : ContMDiffWithinAt I I' (↑n) f s x
      v : Set E
      hmem : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (extChart …
      hsub : HasSubset.Subset v (Inter.inter (extChartAt I x).target (Set.preimage ( …
      hv : ContDiffOn 𝕜 (↑↑n) (Function.comp (↑(extChartAt I' (f x))) (Function.comp …
      hxs' : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(ex …
      hv₁ : HasSubset.Subset (Set.image (↑(extChartAt I x).symm) v) (extChartAt I x) …
      hv₂ : Set.MapsTo f (Set.image (↑(extChartAt I x).symm) v) (extChartAt I' (f x) …
      ⊢ ContMDiffOn I I' (↑n) f (Set.image (↑(extChartAt I x).symm) v)
    -/
    rwa [contMDiffOn_iff_of_subset_source' hv₁ hv₂, PartialEquiv.image_symm_image_of_subset_target]
    /-
      case intro.intro.intro.refine_2.h
      𝕜 : Type u_1
      inst✝¹² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝⁹ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝⁸ : TopologicalSpace M
      inst✝⁷ : ChartedSpace H M
      E' : Type u_5
      inst✝⁶ : NormedAddCommGroup E'
      inst✝⁵ : NormedSpace 𝕜 E'
      H' : Type u_6
      inst✝⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E' H'
      M' : Type u_7
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      f : M → M'
      s✝ : Set M
      x : M
      inst✝¹ : SmoothManifoldWithCorners I M
      inst✝ : SmoothManifoldWithCorners I' M'
      n : Nat
      s : Set M
      hxs : Membership.mem s x
      h : ContMDiffWithinAt I I' (↑n) f s x
      v : Set E
      hmem : Membership.mem (nhdsWithin (↑(extChartAt I x) x) (Inter.inter (extChart …
      hsub : HasSubset.Subset v (Inter.inter (extChartAt I x).target (Set.preimage ( …
      hv : ContDiffOn 𝕜 (↑↑n) (Function.comp (↑(extChartAt I' (f x))) (Function.comp …
      hxs' : Membership.mem (Inter.inter (extChartAt I x).target (Set.preimage (↑(ex …
      hv₁ : HasSubset.Subset (Set.image (↑(extChartAt I x).symm) v) (extChartAt I x) …
      hv₂ : Set.MapsTo f (Set.image (↑(extChartAt I x).symm) v) (extChartAt I' (f x) …
      ⊢ HasSubset.Subset v (extChartAt I x).target
    -/
    exact hsub.trans inter_subset_left
    /-
      🎉 no goals
    -/


/-- If a function is `C^m` within a set at a point, for some finite `m`, then it is `C^m` within
this set on an open set around the basepoint.
-/
theorem ContMDiffWithinAt.contMDiffOn'
    [SmoothManifoldWithCorners I M] [SmoothManifoldWithCorners I' M']
    {m : ℕ} (hm : (m : ℕ∞) ≤ n)
    (h : ContMDiffWithinAt I I' n f s x) :
    ∃ u, IsOpen u ∧ x ∈ u ∧ ContMDiffOn I I' m f (insert x s ∩ u) := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    m : Nat
    hm : LE.le (↑m) n
    h : ContMDiffWithinAt I I' n f s x
    ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (ContMDiffOn I I' ( …
  -/
  rcases contMDiffWithinAt_iff_contMDiffOn_nhds.1 (h.of_le hm) with ⟨t, ht, h't⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    m : Nat
    hm : LE.le (↑m) n
    h : ContMDiffWithinAt I I' n f s x
    t : Set M
    ht : Membership.mem (nhdsWithin x (Insert.insert x s)) t
    h't : ContMDiffOn I I' (↑m) f t
    ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (ContMDiffOn I I' ( …
  -/
  rcases mem_nhdsWithin.1 ht with ⟨u, u_open, xu, hu⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    m : Nat
    hm : LE.le (↑m) n
    h : ContMDiffWithinAt I I' n f s x
    t : Set M
    ht : Membership.mem (nhdsWithin x (Insert.insert x s)) t
    h't : ContMDiffOn I I' (↑m) f t
    u : Set M
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : HasSubset.Subset (Inter.inter u (Insert.insert x s)) t
    ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (ContMDiffOn I I' ( …
  -/
  rw [inter_comm] at hu
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    m : Nat
    hm : LE.le (↑m) n
    h : ContMDiffWithinAt I I' n f s x
    t : Set M
    ht : Membership.mem (nhdsWithin x (Insert.insert x s)) t
    h't : ContMDiffOn I I' (↑m) f t
    u : Set M
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : HasSubset.Subset (Inter.inter (Insert.insert x s) u) t
    ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (ContMDiffOn I I' ( …
  -/
  exact ⟨u, u_open, xu, h't.mono hu⟩
  /-
    🎉 no goals
  -/


/-- If a function is `C^m` within a set at a point, for some finite `m`, then it is `C^m` within
this set on a neighborhood of the basepoint. -/
theorem ContMDiffWithinAt.contMDiffOn
    [SmoothManifoldWithCorners I M] [SmoothManifoldWithCorners I' M']
    {m : ℕ} (hm : (m : ℕ∞) ≤ n)
    (h : ContMDiffWithinAt I I' n f s x) :
    ∃ u ∈ 𝓝[insert x s] x, u ⊆ insert x s ∧ ContMDiffOn I I' m f u := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    m : Nat
    hm : LE.le (↑m) n
    h : ContMDiffWithinAt I I' n f s x
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
  -/
  let ⟨_u, uo, xu, h⟩ := h.contMDiffOn' hm
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    n : ENat
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    m : Nat
    hm : LE.le (↑m) n
    h✝ : ContMDiffWithinAt I I' n f s x
    _u : Set M
    uo : IsOpen _u
    xu : Membership.mem _u x
    h : ContMDiffOn I I' (↑m) f (Inter.inter (Insert.insert x s) _u)
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
  -/
  exact ⟨_, inter_mem_nhdsWithin _ (uo.mem_nhds xu), inter_subset_left, h⟩
  /-
    🎉 no goals
  -/


/-- A function is `C^n` at a point, for `n : ℕ`, if and only if it is `C^n` on
a neighborhood of this point. -/
theorem contMDiffAt_iff_contMDiffOn_nhds
    [SmoothManifoldWithCorners I M] [SmoothManifoldWithCorners I' M'] {n : ℕ} :
    ContMDiffAt I I' n f x ↔ ∃ u ∈ 𝓝 x, ContMDiffOn I I' n f u := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    ⊢ Iff (ContMDiffAt I I' (↑n) f x) (Exists fun u => And (Membership.mem (nhds x …
  -/
  simp [← contMDiffWithinAt_univ, contMDiffWithinAt_iff_contMDiffOn_nhds, nhdsWithin_univ]
  /-
    🎉 no goals
  -/


/-- Note: This does not hold for `n = ∞`. `f` being `C^∞` at `x` means that for every `n`, `f` is
`C^n` on some neighborhood of `x`, but this neighborhood can depend on `n`. -/
theorem contMDiffAt_iff_contMDiffAt_nhds
    [SmoothManifoldWithCorners I M] [SmoothManifoldWithCorners I' M'] {n : ℕ} :
    ContMDiffAt I I' n f x ↔ ∀ᶠ x' in 𝓝 x, ContMDiffAt I I' n f x' := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    ⊢ Iff (ContMDiffAt I I' (↑n) f x) (Filter.Eventually (fun x' => ContMDiffAt I  …
  -/
  refine ⟨?_, fun h => h.self_of_nhds⟩
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    ⊢ ContMDiffAt I I' (↑n) f x → Filter.Eventually (fun x' => ContMDiffAt I I' (↑ …
  -/
  rw [contMDiffAt_iff_contMDiffOn_nhds]
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    ⊢ (Exists fun u => And (Membership.mem (nhds x) u) (ContMDiffOn I I' (↑n) f u) …
  -/
  rintro ⟨u, hu, h⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    u : Set M
    hu : Membership.mem (nhds x) u
    h : ContMDiffOn I I' (↑n) f u
    ⊢ Filter.Eventually (fun x' => ContMDiffAt I I' (↑n) f x') (nhds x)
  -/
  refine (eventually_mem_nhds_iff.mpr hu).mono fun x' hx' => ?_
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    u : Set M
    hu : Membership.mem (nhds x) u
    h : ContMDiffOn I I' (↑n) f u
    x' : M
    hx' : Membership.mem (nhds x') u
    ⊢ ContMDiffAt I I' (↑n) f x'
  -/
  exact (h x' <| mem_of_mem_nhds hx').contMDiffAt hx'
  /-
    🎉 no goals
  -/


/-- Note: This does not hold for `n = ∞`. `f` being `C^∞` at `x` means that for every `n`, `f` is
`C^n` on some neighborhood of `x`, but this neighborhood can depend on `n`. -/
theorem contMDiffWithinAt_iff_contMDiffWithinAt_nhdsWithin
    [SmoothManifoldWithCorners I M] [SmoothManifoldWithCorners I' M'] {n : ℕ} :
    ContMDiffWithinAt I I' n f s x ↔
      ∀ᶠ x' in 𝓝[insert x s] x, ContMDiffWithinAt I I' n f s x' := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    ⊢ Iff (ContMDiffWithinAt I I' (↑n) f s x) (Filter.Eventually (fun x' => ContMD …
  -/
  refine ⟨?_, fun h ↦ mem_of_mem_nhdsWithin (mem_insert x s) h⟩
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    ⊢ ContMDiffWithinAt I I' (↑n) f s x → Filter.Eventually (fun x' => ContMDiffWi …
  -/
  rw [contMDiffWithinAt_iff_contMDiffOn_nhds]
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    ⊢ (Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) ( …
  -/
  rintro ⟨u, hu, h⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    u : Set M
    hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
    h : ContMDiffOn I I' (↑n) f u
    ⊢ Filter.Eventually (fun x' => ContMDiffWithinAt I I' (↑n) f s x') (nhdsWithin …
  -/
  filter_upwards [hu, eventually_mem_nhdsWithin_iff.mpr hu] with x' h'x' hx'
  /-
    case h
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    u : Set M
    hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
    h : ContMDiffOn I I' (↑n) f u
    x' : M
    h'x' : Membership.mem u x'
    hx' : Membership.mem (nhdsWithin x' (Insert.insert x s)) u
    ⊢ ContMDiffWithinAt I I' (↑n) f s x'
  -/
  apply (h x' h'x').mono_of_mem_nhdsWithin
  /-
    case h
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    H : Type u_3
    inst✝⁹ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_4
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    E' : Type u_5
    inst✝⁶ : NormedAddCommGroup E'
    inst✝⁵ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M' : Type u_7
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    f : M → M'
    s : Set M
    x : M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SmoothManifoldWithCorners I' M'
    n : Nat
    u : Set M
    hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
    h : ContMDiffOn I I' (↑n) f u
    x' : M
    h'x' : Membership.mem u x'
    hx' : Membership.mem (nhdsWithin x' (Insert.insert x s)) u
    ⊢ Membership.mem (nhdsWithin x' s) u
  -/
  exact nhdsWithin_mono _ (subset_insert x s) hx'
  /-
    🎉 no goals
  -/


theorem ContMDiffWithinAt.congr (h : ContMDiffWithinAt I I' n f s x) (h₁ : ∀ y ∈ s, f₁ y = f y)
    (hx : f₁ x = f x) : ContMDiffWithinAt I I' n f₁ s x :=
  (contDiffWithinAt_localInvariantProp n).liftPropWithinAt_congr h h₁ hx


theorem contMDiffWithinAt_congr (h₁ : ∀ y ∈ s, f₁ y = f y) (hx : f₁ x = f x) :
    ContMDiffWithinAt I I' n f₁ s x ↔ ContMDiffWithinAt I I' n f s x :=
  (contDiffWithinAt_localInvariantProp n).liftPropWithinAt_congr_iff h₁ hx


theorem ContMDiffWithinAt.congr_of_mem
    (h : ContMDiffWithinAt I I' n f s x) (h₁ : ∀ y ∈ s, f₁ y = f y) (hx : x ∈ s) :
    ContMDiffWithinAt I I' n f₁ s x :=
  (contDiffWithinAt_localInvariantProp n).liftPropWithinAt_congr_of_mem h h₁ hx


theorem contMDiffWithinAt_congr_of_mem (h₁ : ∀ y ∈ s, f₁ y = f y) (hx : x ∈ s) :
    ContMDiffWithinAt I I' n f₁ s x ↔ ContMDiffWithinAt I I' n f s x :=
  (contDiffWithinAt_localInvariantProp n).liftPropWithinAt_congr_iff_of_mem h₁ hx


theorem ContMDiffWithinAt.congr_of_eventuallyEq (h : ContMDiffWithinAt I I' n f s x)
    (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) : ContMDiffWithinAt I I' n f₁ s x :=
  (contDiffWithinAt_localInvariantProp n).liftPropWithinAt_congr_of_eventuallyEq h h₁ hx


theorem ContMDiffWithinAt.congr_of_eventuallyEq_of_mem (h : ContMDiffWithinAt I I' n f s x)
    (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : x ∈ s) : ContMDiffWithinAt I I' n f₁ s x :=
  (contDiffWithinAt_localInvariantProp n).liftPropWithinAt_congr_of_eventuallyEq_of_mem h h₁ hx


theorem Filter.EventuallyEq.contMDiffWithinAt_iff (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) :
    ContMDiffWithinAt I I' n f₁ s x ↔ ContMDiffWithinAt I I' n f s x :=
  (contDiffWithinAt_localInvariantProp n).liftPropWithinAt_congr_iff_of_eventuallyEq h₁ hx


theorem ContMDiffAt.congr_of_eventuallyEq (h : ContMDiffAt I I' n f x) (h₁ : f₁ =ᶠ[𝓝 x] f) :
    ContMDiffAt I I' n f₁ x :=
  (contDiffWithinAt_localInvariantProp n).liftPropAt_congr_of_eventuallyEq h h₁


theorem Filter.EventuallyEq.contMDiffAt_iff (h₁ : f₁ =ᶠ[𝓝 x] f) :
    ContMDiffAt I I' n f₁ x ↔ ContMDiffAt I I' n f x :=
  (contDiffWithinAt_localInvariantProp n).liftPropAt_congr_iff_of_eventuallyEq h₁


theorem ContMDiffOn.congr (h : ContMDiffOn I I' n f s) (h₁ : ∀ y ∈ s, f₁ y = f y) :
    ContMDiffOn I I' n f₁ s :=
  (contDiffWithinAt_localInvariantProp n).liftPropOn_congr h h₁


theorem contMDiffOn_congr (h₁ : ∀ y ∈ s, f₁ y = f y) :
    ContMDiffOn I I' n f₁ s ↔ ContMDiffOn I I' n f s :=
  (contDiffWithinAt_localInvariantProp n).liftPropOn_congr_iff h₁


theorem ContMDiffOn.congr_mono (hf : ContMDiffOn I I' n f s) (h₁ : ∀ y ∈ s₁, f₁ y = f y)
    (hs : s₁ ⊆ s) : ContMDiffOn I I' n f₁ s₁ :=
  (hf.mono hs).congr h₁


/-- Being `C^n` is a local property. -/
theorem contMDiffOn_of_locally_contMDiffOn
    (h : ∀ x ∈ s, ∃ u, IsOpen u ∧ x ∈ u ∧ ContMDiffOn I I' n f (s ∩ u)) : ContMDiffOn I I' n f s :=
  (contDiffWithinAt_localInvariantProp n).liftPropOn_of_locally_liftPropOn h


theorem contMDiff_of_locally_contMDiffOn (h : ∀ x, ∃ u, IsOpen u ∧ x ∈ u ∧ ContMDiffOn I I' n f u) :
    ContMDiff I I' n f :=
  (contDiffWithinAt_localInvariantProp n).liftProp_of_locally_liftPropOn h

