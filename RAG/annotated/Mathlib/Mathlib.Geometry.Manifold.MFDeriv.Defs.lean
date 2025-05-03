variable (I I') in
/-- Property in the model space of a model with corners of being differentiable within at set at a
point, when read in the model vector space. This property will be lifted to manifolds to define
differentiable functions between manifolds. -/
def DifferentiableWithinAtProp (f : H → H') (s : Set H) (x : H) : Prop :=
  DifferentiableWithinAt 𝕜 (I' ∘ f ∘ I.symm) (I.symm ⁻¹' s ∩ Set.range I) (I x)


theorem differentiableWithinAtProp_self_source {f : E → H'} {s : Set E} {x : E} :
    DifferentiableWithinAtProp 𝓘(𝕜, E) I' f s x ↔ DifferentiableWithinAt 𝕜 (I' ∘ f) s x := by
  simp_rw [DifferentiableWithinAtProp, modelWithCornersSelf_coe, range_id, inter_univ,
    modelWithCornersSelf_coe_symm, CompTriple.comp_eq, preimage_id_eq, id_eq]


theorem DifferentiableWithinAtProp_self {f : E → E'} {s : Set E} {x : E} :
    DifferentiableWithinAtProp 𝓘(𝕜, E) 𝓘(𝕜, E') f s x ↔ DifferentiableWithinAt 𝕜 f s x :=
  differentiableWithinAtProp_self_source


theorem differentiableWithinAtProp_self_target {f : H → E'} {s : Set H} {x : H} :
    DifferentiableWithinAtProp I 𝓘(𝕜, E') f s x ↔
      DifferentiableWithinAt 𝕜 (f ∘ I.symm) (I.symm ⁻¹' s ∩ range I) (I x) :=
  Iff.rfl


/-- Being differentiable in the model space is a local property, invariant under smooth maps.
Therefore, it will lift nicely to manifolds. -/
theorem differentiableWithinAt_localInvariantProp :
    (contDiffGroupoid ∞ I).LocalInvariantProp (contDiffGroupoid ∞ I')
      (DifferentiableWithinAtProp I I') :=
  { is_local := by
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
        ⊢ ∀ {s : Set H} {x : H} {u : Set H} {f : H → H'}, IsOpen u → Membership.mem u  …
      -/
      intro s x u f u_open xu
      have : I.symm ⁻¹' (s ∩ u) ∩ Set.range I = I.symm ⁻¹' s ∩ Set.range I ∩ I.symm ⁻¹' u := by
        simp only [Set.inter_right_comm, Set.preimage_inter]
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
        s : Set H
        x : H
        u : Set H
        f : H → H'
        u_open : IsOpen u
        xu : Membership.mem u x
        this : Eq (Inter.inter (Set.preimage (↑I.symm) (Inter.inter s u)) (Set.range ↑ …
        ⊢ Iff (DifferentiableWithinAtProp I I' f s x) (DifferentiableWithinAtProp I I' …
      -/
      rw [DifferentiableWithinAtProp, DifferentiableWithinAtProp, this]
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
        s : Set H
        x : H
        u : Set H
        f : H → H'
        u_open : IsOpen u
        xu : Membership.mem u x
        this : Eq (Inter.inter (Set.preimage (↑I.symm) (Inter.inter s u)) (Set.range ↑ …
        ⊢ Iff (DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp f ↑I.symm) …
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
        s : Set H
        x : H
        u : Set H
        f : H → H'
        u_open : IsOpen u
        xu : Membership.mem u x
        this : Eq (Inter.inter (Set.preimage (↑I.symm) (Inter.inter s u)) (Set.range ↑ …
        ⊢ Iff (DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp f ↑I.symm) …
      -/
      apply differentiableWithinAt_inter
      have : u ∈ 𝓝 (I.symm (I x)) := by
        rw [ModelWithCorners.left_inv]
        exact u_open.mem_nhds xu
      /-
        case ht
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
      apply I.continuous_symm.continuousAt this
      /-
        🎉 no goals
      -/
    right_invariance' := by
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
        ⊢ ∀ {s : Set H} {x : H} {f : H → H'} {e : PartialHomeomorph H H}, Membership.m …
      -/
      intro s x f e he hx h
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
        s : Set H
        x : H
        f : H → H'
        e : PartialHomeomorph H H
        he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
        hx : Membership.mem e.source x
        h : DifferentiableWithinAtProp I I' f s x
        ⊢ DifferentiableWithinAtProp I I' (Function.comp f ↑e.symm) (Set.preimage (↑e. …
      -/
      rw [DifferentiableWithinAtProp] at h ⊢
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
        s : Set H
        x : H
        f : H → H'
        e : PartialHomeomorph H H
        he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
        hx : Membership.mem e.source x
        h : DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp f ↑I.symm)) ( …
        ⊢ DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp (Function.comp  …
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
        s : Set H
        x : H
        f : H → H'
        e : PartialHomeomorph H H
        he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
        hx : Membership.mem e.source x
        h : DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp f ↑I.symm)) ( …
        this : Eq (↑I x) (Function.comp (↑I) (Function.comp ↑e.symm ↑I.symm) (↑I (↑e x …
        ⊢ DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp (Function.comp  …
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
        s : Set H
        x : H
        f : H → H'
        e : PartialHomeomorph H H
        he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
        hx : Membership.mem e.source x
        h : DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp f ↑I.symm)) ( …
        this : Eq (↑I x) (Function.comp (↑I) (Function.comp ↑e.symm ↑I.symm) (↑I (↑e x …
        ⊢ DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp (Function.comp  …
      -/
      have : I (e x) ∈ I.symm ⁻¹' e.target ∩ Set.range I := by simp only [hx, mfld_simps]
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
        s : Set H
        x : H
        f : H → H'
        e : PartialHomeomorph H H
        he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
        hx : Membership.mem e.source x
        h : DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp f ↑I.symm)) ( …
        this✝ : Eq (↑I x) (Function.comp (↑I) (Function.comp ↑e.symm ↑I.symm) (↑I (↑e  …
        this : Membership.mem (Inter.inter (Set.preimage (↑I.symm) e.target) (Set.rang …
        ⊢ DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp (Function.comp  …
      -/
      have := (mem_groupoid_of_pregroupoid.2 he).2.contDiffWithinAt this
      convert (h.comp' _ (this.differentiableWithinAt (mod_cast le_top))).mono_of_mem_nhdsWithin _
        using 1
        /-
          case h.e'_11
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
          s : Set H
          x : H
          f : H → H'
          e : PartialHomeomorph H H
          he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
          hx : Membership.mem e.source x
          h : DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp f ↑I.symm)) ( …
          this✝¹ : Eq (↑I x) (Function.comp (↑I) (Function.comp ↑e.symm ↑I.symm) (↑I (↑e …
          this✝ : Membership.mem (Inter.inter (Set.preimage (↑I.symm) e.target) (Set.ran …
          this : ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑e.sym …
          ⊢ Eq (Function.comp (↑I') (Function.comp (Function.comp f ↑e.symm) ↑I.symm)) ( …
        -/
      · ext y; simp only [mfld_simps]
               /-
                 🎉 no goals
               -/
      refine
        mem_nhdsWithin.mpr
          ⟨I.symm ⁻¹' e.target, e.open_target.preimage I.continuous_symm, by
            simp_rw [Set.mem_preimage, I.left_inv, e.mapsTo hx], ?_⟩
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
        s : Set H
        x : H
        f : H → H'
        e : PartialHomeomorph H H
        he : Membership.mem (contDiffGroupoid (↑Top.top) I) e
        hx : Membership.mem e.source x
        h : DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp f ↑I.symm)) ( …
        this✝¹ : Eq (↑I x) (Function.comp (↑I) (Function.comp ↑e.symm ↑I.symm) (↑I (↑e …
        this✝ : Membership.mem (Inter.inter (Set.preimage (↑I.symm) e.target) (Set.ran …
        this : ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I) (Function.comp ↑e.sym …
        ⊢ HasSubset.Subset (Inter.inter (Set.preimage (↑I.symm) e.target) (Inter.inter …
      -/
      mfld_set_tac
      /-
        🎉 no goals
      -/
    congr_of_forall := by
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
        ⊢ ∀ {s : Set H} {x : H} {f g : H → H'}, (∀ (y : H), Membership.mem s y → Eq (f …
      -/
      intro s x f g h hx hf
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
        s : Set H
        x : H
        f g : H → H'
        h : ∀ (y : H), Membership.mem s y → Eq (f y) (g y)
        hx : Eq (f x) (g x)
        hf : DifferentiableWithinAtProp I I' f s x
        ⊢ DifferentiableWithinAtProp I I' g s x
      -/
      apply hf.congr
        /-
          case ht
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
          s : Set H
          x : H
          f g : H → H'
          h : ∀ (y : H), Membership.mem s y → Eq (f y) (g y)
          hx : Eq (f x) (g x)
          hf : DifferentiableWithinAtProp I I' f s x
          ⊢ ∀ (x : E), Membership.mem (Inter.inter (Set.preimage (↑I.symm) s) (Set.range …
        -/
      · intro y hy
        /-
          case ht
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
          s : Set H
          x : H
          f g : H → H'
          h : ∀ (y : H), Membership.mem s y → Eq (f y) (g y)
          hx : Eq (f x) (g x)
          hf : DifferentiableWithinAtProp I I' f s x
          y : E
          hy : Membership.mem (Inter.inter (Set.preimage (↑I.symm) s) (Set.range ↑I)) y
          ⊢ Eq (Function.comp (↑I') (Function.comp g ↑I.symm) y) (Function.comp (↑I') (F …
        -/
        simp only [mfld_simps] at hy
        /-
          case ht
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
          s : Set H
          x : H
          f g : H → H'
          h : ∀ (y : H), Membership.mem s y → Eq (f y) (g y)
          hx : Eq (f x) (g x)
          hf : DifferentiableWithinAtProp I I' f s x
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
          s : Set H
          x : H
          f g : H → H'
          h : ∀ (y : H), Membership.mem s y → Eq (f y) (g y)
          hx : Eq (f x) (g x)
          hf : DifferentiableWithinAtProp I I' f s x
          ⊢ Eq (Function.comp (↑I') (Function.comp g ↑I.symm) (↑I x)) (Function.comp (↑I …
        -/
      · simp only [hx, mfld_simps]
        /-
          🎉 no goals
        -/
    left_invariance' := by
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
        ⊢ ∀ {s : Set H} {x : H} {f : H → H'} {e' : PartialHomeomorph H' H'}, Membershi …
      -/
      intro s x f e' he' hs hx h
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
        s : Set H
        x : H
        f : H → H'
        e' : PartialHomeomorph H' H'
        he' : Membership.mem (contDiffGroupoid (↑Top.top) I') e'
        hs : HasSubset.Subset s (Set.preimage f e'.source)
        hx : Membership.mem e'.source (f x)
        h : DifferentiableWithinAtProp I I' f s x
        ⊢ DifferentiableWithinAtProp I I' (Function.comp (↑e') f) s x
      -/
      rw [DifferentiableWithinAtProp] at h ⊢
      have A : (I' ∘ f ∘ I.symm) (I x) ∈ I'.symm ⁻¹' e'.source ∩ Set.range I' := by
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
        s : Set H
        x : H
        f : H → H'
        e' : PartialHomeomorph H' H'
        he' : Membership.mem (contDiffGroupoid (↑Top.top) I') e'
        hs : HasSubset.Subset s (Set.preimage f e'.source)
        hx : Membership.mem e'.source (f x)
        h : DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp f ↑I.symm)) ( …
        A : Membership.mem (Inter.inter (Set.preimage (↑I'.symm) e'.source) (Set.range …
        ⊢ DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp (Function.comp  …
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
        s : Set H
        x : H
        f : H → H'
        e' : PartialHomeomorph H' H'
        he' : Membership.mem (contDiffGroupoid (↑Top.top) I') e'
        hs : HasSubset.Subset s (Set.preimage f e'.source)
        hx : Membership.mem e'.source (f x)
        h : DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp f ↑I.symm)) ( …
        A : Membership.mem (Inter.inter (Set.preimage (↑I'.symm) e'.source) (Set.range …
        this : ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I') (Function.comp ↑e' ↑ …
        ⊢ DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp (Function.comp  …
      -/
      convert (this.differentiableWithinAt (mod_cast le_top)).comp _ h _
        /-
          case h.e'_11
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
          s : Set H
          x : H
          f : H → H'
          e' : PartialHomeomorph H' H'
          he' : Membership.mem (contDiffGroupoid (↑Top.top) I') e'
          hs : HasSubset.Subset s (Set.preimage f e'.source)
          hx : Membership.mem e'.source (f x)
          h : DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp f ↑I.symm)) ( …
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
          s : Set H
          x : H
          f : H → H'
          e' : PartialHomeomorph H' H'
          he' : Membership.mem (contDiffGroupoid (↑Top.top) I') e'
          hs : HasSubset.Subset s (Set.preimage f e'.source)
          hx : Membership.mem e'.source (f x)
          h : DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp f ↑I.symm)) ( …
          A : Membership.mem (Inter.inter (Set.preimage (↑I'.symm) e'.source) (Set.range …
          this : ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑I') (Function.comp ↑e' ↑ …
          ⊢ Set.MapsTo (Function.comp (↑I') (Function.comp f ↑I.symm)) (Inter.inter (Set …
        -/
      · intro y hy; simp only [mfld_simps] at hy; simpa only [hy, mfld_simps] using hs hy.1 }
                                                  /-
                                                    🎉 no goals
                                                  -/


@[deprecated (since := "2024-10-10")]
alias differentiable_within_at_localInvariantProp := differentiableWithinAt_localInvariantProp


variable (I) in
/-- Predicate ensuring that, at a point and within a set, a function can have at most one
derivative. This is expressed using the preferred chart at the considered point. -/
def UniqueMDiffWithinAt (s : Set M) (x : M) :=
  UniqueDiffWithinAt 𝕜 ((extChartAt I x).symm ⁻¹' s ∩ range I) ((extChartAt I x) x)


variable (I) in
/-- Predicate ensuring that, at all points of a set, a function can have at most one derivative. -/
def UniqueMDiffOn (s : Set M) :=
  ∀ x ∈ s, UniqueMDiffWithinAt I s x


variable (I I') in
/-- `MDifferentiableWithinAt I I' f s x` indicates that the function `f` between manifolds
has a derivative at the point `x` within the set `s`.
This is a generalization of `DifferentiableWithinAt` to manifolds.

We require continuity in the definition, as otherwise points close to `x` in `s` could be sent by
`f` outside of the chart domain around `f x`. Then the chart could do anything to the image points,
and in particular by coincidence `writtenInExtChartAt I I' x f` could be differentiable, while
this would not mean anything relevant. -/
def MDifferentiableWithinAt (f : M → M') (s : Set M) (x : M) :=
  LiftPropWithinAt (DifferentiableWithinAtProp I I') f s x


theorem mdifferentiableWithinAt_iff' (f : M → M') (s : Set M) (x : M) :
    MDifferentiableWithinAt I I' f s x ↔ ContinuousWithinAt f s x ∧
    DifferentiableWithinAt 𝕜 (writtenInExtChartAt I I' x f)
      ((extChartAt I x).symm ⁻¹' s ∩ range I) ((extChartAt I x) x) := by
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
    ⊢ Iff (MDifferentiableWithinAt I I' f s x) (And (ContinuousWithinAt f s x) (Di …
  -/
  rw [MDifferentiableWithinAt, liftPropWithinAt_iff']; rfl
                                                       /-
                                                         🎉 no goals
                                                       -/


@[deprecated (since := "2024-04-30")]
alias mdifferentiableWithinAt_iff_liftPropWithinAt := mdifferentiableWithinAt_iff'


theorem MDifferentiableWithinAt.continuousWithinAt {f : M → M'} {s : Set M} {x : M}
    (hf : MDifferentiableWithinAt I I' f s x) :
    ContinuousWithinAt f s x :=
  mdifferentiableWithinAt_iff' .. |>.1 hf |>.1


theorem MDifferentiableWithinAt.differentiableWithinAt_writtenInExtChartAt
    {f : M → M'} {s : Set M} {x : M} (hf : MDifferentiableWithinAt I I' f s x) :
    DifferentiableWithinAt 𝕜 (writtenInExtChartAt I I' x f)
      ((extChartAt I x).symm ⁻¹' s ∩ range I) ((extChartAt I x) x) :=
  mdifferentiableWithinAt_iff' .. |>.1 hf |>.2


variable (I I') in
/-- `MDifferentiableAt I I' f x` indicates that the function `f` between manifolds
has a derivative at the point `x`.
This is a generalization of `DifferentiableAt` to manifolds.

We require continuity in the definition, as otherwise points close to `x` could be sent by
`f` outside of the chart domain around `f x`. Then the chart could do anything to the image points,
and in particular by coincidence `writtenInExtChartAt I I' x f` could be differentiable, while
this would not mean anything relevant. -/
def MDifferentiableAt (f : M → M') (x : M) :=
  LiftPropAt (DifferentiableWithinAtProp I I') f x


theorem mdifferentiableAt_iff (f : M → M') (x : M) :
    MDifferentiableAt I I' f x ↔ ContinuousAt f x ∧
    DifferentiableWithinAt 𝕜 (writtenInExtChartAt I I' x f) (range I) ((extChartAt I x) x) := by
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
    x : M
    ⊢ Iff (MDifferentiableAt I I' f x) (And (ContinuousAt f x) (DifferentiableWith …
  -/
  rw [MDifferentiableAt, liftPropAt_iff]
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
    x : M
    ⊢ Iff (And (ContinuousAt f x) (DifferentiableWithinAtProp I I' (Function.comp  …
  -/
  congrm _ ∧ ?_
  /-
    case a
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
    x : M
    ⊢ Iff (DifferentiableWithinAtProp I I' (Function.comp (↑(chartAt H' (f x))) (F …
  -/
  simp [DifferentiableWithinAtProp, Set.univ_inter]
  -- Porting note: `rfl` wasn't needed
  /-
    case a
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
    x : M
    ⊢ Iff (DifferentiableWithinAt 𝕜 (Function.comp (↑I') (Function.comp (Function. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-30")]
alias mdifferentiableAt_iff_liftPropAt := mdifferentiableAt_iff


theorem MDifferentiableAt.continuousAt {f : M → M'} {x : M} (hf : MDifferentiableAt I I' f x) :
    ContinuousAt f x :=
  mdifferentiableAt_iff .. |>.1 hf |>.1


theorem MDifferentiableAt.differentiableWithinAt_writtenInExtChartAt {f : M → M'} {x : M}
    (hf : MDifferentiableAt I I' f x) :
    DifferentiableWithinAt 𝕜 (writtenInExtChartAt I I' x f) (range I) ((extChartAt I x) x) :=
  mdifferentiableAt_iff .. |>.1 hf |>.2


variable (I I') in
/-- `MDifferentiableOn I I' f s` indicates that the function `f` between manifolds
has a derivative within `s` at all points of `s`.
This is a generalization of `DifferentiableOn` to manifolds. -/
def MDifferentiableOn (f : M → M') (s : Set M) :=
  ∀ x ∈ s, MDifferentiableWithinAt I I' f s x


variable (I I') in
/-- `MDifferentiable I I' f` indicates that the function `f` between manifolds
has a derivative everywhere.
This is a generalization of `Differentiable` to manifolds. -/
def MDifferentiable (f : M → M') :=
  ∀ x, MDifferentiableAt I I' f x


variable (I I') in
/-- Prop registering if a partial homeomorphism is a local diffeomorphism on its source -/
def PartialHomeomorph.MDifferentiable (f : PartialHomeomorph M M') :=
  MDifferentiableOn I I' f f.source ∧ MDifferentiableOn I' I f.symm f.target


variable (I I') in
/-- `HasMFDerivWithinAt I I' f s x f'` indicates that the function `f` between manifolds
has, at the point `x` and within the set `s`, the derivative `f'`. Here, `f'` is a continuous linear
map from the tangent space at `x` to the tangent space at `f x`.

This is a generalization of `HasFDerivWithinAt` to manifolds (as indicated by the prefix `m`).
The order of arguments is changed as the type of the derivative `f'` depends on the choice of `x`.

We require continuity in the definition, as otherwise points close to `x` in `s` could be sent by
`f` outside of the chart domain around `f x`. Then the chart could do anything to the image points,
and in particular by coincidence `writtenInExtChartAt I I' x f` could be differentiable, while
this would not mean anything relevant. -/
def HasMFDerivWithinAt (f : M → M') (s : Set M) (x : M)
    (f' : TangentSpace I x →L[𝕜] TangentSpace I' (f x)) :=
  ContinuousWithinAt f s x ∧
    HasFDerivWithinAt (writtenInExtChartAt I I' x f : E → E') f'
      ((extChartAt I x).symm ⁻¹' s ∩ range I) ((extChartAt I x) x)


variable (I I') in
/-- `HasMFDerivAt I I' f x f'` indicates that the function `f` between manifolds
has, at the point `x`, the derivative `f'`. Here, `f'` is a continuous linear
map from the tangent space at `x` to the tangent space at `f x`.

We require continuity in the definition, as otherwise points close to `x` `s` could be sent by
`f` outside of the chart domain around `f x`. Then the chart could do anything to the image points,
and in particular by coincidence `writtenInExtChartAt I I' x f` could be differentiable, while
this would not mean anything relevant. -/
def HasMFDerivAt (f : M → M') (x : M) (f' : TangentSpace I x →L[𝕜] TangentSpace I' (f x)) :=
  ContinuousAt f x ∧
    HasFDerivWithinAt (writtenInExtChartAt I I' x f : E → E') f' (range I) ((extChartAt I x) x)


open Classical in
variable (I I') in
/-- Let `f` be a function between two smooth manifolds. Then `mfderivWithin I I' f s x` is the
derivative of `f` at `x` within `s`, as a continuous linear map from the tangent space at `x` to the
tangent space at `f x`. -/
def mfderivWithin (f : M → M') (s : Set M) (x : M) : TangentSpace I x →L[𝕜] TangentSpace I' (f x) :=
  if MDifferentiableWithinAt I I' f s x then
    (fderivWithin 𝕜 (writtenInExtChartAt I I' x f) ((extChartAt I x).symm ⁻¹' s ∩ range I)
        ((extChartAt I x) x) :
      _)
  else 0


open Classical in
variable (I I') in
/-- Let `f` be a function between two smooth manifolds. Then `mfderiv I I' f x` is the derivative of
`f` at `x`, as a continuous linear map from the tangent space at `x` to the tangent space at
`f x`. -/
def mfderiv (f : M → M') (x : M) : TangentSpace I x →L[𝕜] TangentSpace I' (f x) :=
  if MDifferentiableAt I I' f x then
    (fderivWithin 𝕜 (writtenInExtChartAt I I' x f : E → E') (range I) ((extChartAt I x) x) : _)
  else 0


variable (I I') in
/-- The derivative within a set, as a map between the tangent bundles -/
def tangentMapWithin (f : M → M') (s : Set M) : TangentBundle I M → TangentBundle I' M' := fun p =>
  ⟨f p.1, (mfderivWithin I I' f s p.1 : TangentSpace I p.1 → TangentSpace I' (f p.1)) p.2⟩


variable (I I') in
/-- The derivative, as a map between the tangent bundles -/
def tangentMap (f : M → M') : TangentBundle I M → TangentBundle I' M' := fun p =>
  ⟨f p.1, (mfderiv I I' f p.1 : TangentSpace I p.1 → TangentSpace I' (f p.1)) p.2⟩


