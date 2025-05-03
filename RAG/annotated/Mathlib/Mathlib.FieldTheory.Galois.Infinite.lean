lemma fixingSubgroup_isClosed (L : IntermediateField k K) [IsGalois k K] :
    IsClosed (L.fixingSubgroup : Set (K ≃ₐ[k] K)) where
    isOpen_compl := isOpen_iff_mem_nhds.mpr fun σ h => by
      /-
        k : Type u_1
        K : Type u_2
        inst✝³ : Field k
        inst✝² : Field K
        inst✝¹ : Algebra k K
        L : IntermediateField k K
        inst✝ : IsGalois k K
        σ : AlgEquiv k K K
        h : Membership.mem (HasCompl.compl ↑L.fixingSubgroup) σ
        ⊢ Membership.mem (nhds σ) (HasCompl.compl ↑L.fixingSubgroup)
      -/
      apply mem_nhds_iff.mpr
      /-
        k : Type u_1
        K : Type u_2
        inst✝³ : Field k
        inst✝² : Field K
        inst✝¹ : Algebra k K
        L : IntermediateField k K
        inst✝ : IsGalois k K
        σ : AlgEquiv k K K
        h : Membership.mem (HasCompl.compl ↑L.fixingSubgroup) σ
        ⊢ Exists fun t => And (HasSubset.Subset t (HasCompl.compl ↑L.fixingSubgroup))  …
      -/
      rcases Set.not_subset.mp ((mem_fixingSubgroup_iff (K ≃ₐ[k] K)).not.mp h) with ⟨y, yL, ne⟩
      /-
        case intro.intro
        k : Type u_1
        K : Type u_2
        inst✝³ : Field k
        inst✝² : Field K
        inst✝¹ : Algebra k K
        L : IntermediateField k K
        inst✝ : IsGalois k K
        σ : AlgEquiv k K K
        h : Membership.mem (HasCompl.compl ↑L.fixingSubgroup) σ
        y : K
        yL : Membership.mem (↑L) y
        ne : Not (Membership.mem (fun a => Eq (HSMul.hSMul σ a) a) y)
        ⊢ Exists fun t => And (HasSubset.Subset t (HasCompl.compl ↑L.fixingSubgroup))  …
      -/
      use σ • ((adjoin k {y}).1.fixingSubgroup : Set (K ≃ₐ[k] K))
      /-
        case h
        k : Type u_1
        K : Type u_2
        inst✝³ : Field k
        inst✝² : Field K
        inst✝¹ : Algebra k K
        L : IntermediateField k K
        inst✝ : IsGalois k K
        σ : AlgEquiv k K K
        h : Membership.mem (HasCompl.compl ↑L.fixingSubgroup) σ
        y : K
        yL : Membership.mem (↑L) y
        ne : Not (Membership.mem (fun a => Eq (HSMul.hSMul σ a) a) y)
        ⊢ And (HasSubset.Subset (HSMul.hSMul σ ↑(FiniteGaloisIntermediateField.adjoin  …
      -/
      constructor
        /-
          case h.left
          k : Type u_1
          K : Type u_2
          inst✝³ : Field k
          inst✝² : Field K
          inst✝¹ : Algebra k K
          L : IntermediateField k K
          inst✝ : IsGalois k K
          σ : AlgEquiv k K K
          h : Membership.mem (HasCompl.compl ↑L.fixingSubgroup) σ
          y : K
          yL : Membership.mem (↑L) y
          ne : Not (Membership.mem (fun a => Eq (HSMul.hSMul σ a) a) y)
          ⊢ HasSubset.Subset (HSMul.hSMul σ ↑(FiniteGaloisIntermediateField.adjoin k (Si …
        -/
      · intro f hf
        /-
          case h.left
          k : Type u_1
          K : Type u_2
          inst✝³ : Field k
          inst✝² : Field K
          inst✝¹ : Algebra k K
          L : IntermediateField k K
          inst✝ : IsGalois k K
          σ : AlgEquiv k K K
          h : Membership.mem (HasCompl.compl ↑L.fixingSubgroup) σ
          y : K
          yL : Membership.mem (↑L) y
          ne : Not (Membership.mem (fun a => Eq (HSMul.hSMul σ a) a) y)
          f : AlgEquiv k K K
          hf : Membership.mem (HSMul.hSMul σ ↑(FiniteGaloisIntermediateField.adjoin k (S …
          ⊢ Membership.mem (HasCompl.compl ↑L.fixingSubgroup) f
        -/
        rcases (Set.mem_smul_set.mp hf) with ⟨g, hg, eq⟩
        /-
          case h.left.intro.intro
          k : Type u_1
          K : Type u_2
          inst✝³ : Field k
          inst✝² : Field K
          inst✝¹ : Algebra k K
          L : IntermediateField k K
          inst✝ : IsGalois k K
          σ : AlgEquiv k K K
          h : Membership.mem (HasCompl.compl ↑L.fixingSubgroup) σ
          y : K
          yL : Membership.mem (↑L) y
          ne : Not (Membership.mem (fun a => Eq (HSMul.hSMul σ a) a) y)
          f : AlgEquiv k K K
          hf : Membership.mem (HSMul.hSMul σ ↑(FiniteGaloisIntermediateField.adjoin k (S …
          g : AlgEquiv k K K
          hg : Membership.mem (↑(FiniteGaloisIntermediateField.adjoin k (Singleton.singl …
          eq : Eq (HSMul.hSMul σ g) f
          ⊢ Membership.mem (HasCompl.compl ↑L.fixingSubgroup) f
        -/
        simp only [Set.mem_compl_iff, SetLike.mem_coe, ← eq]
        /-
          case h.left.intro.intro
          k : Type u_1
          K : Type u_2
          inst✝³ : Field k
          inst✝² : Field K
          inst✝¹ : Algebra k K
          L : IntermediateField k K
          inst✝ : IsGalois k K
          σ : AlgEquiv k K K
          h : Membership.mem (HasCompl.compl ↑L.fixingSubgroup) σ
          y : K
          yL : Membership.mem (↑L) y
          ne : Not (Membership.mem (fun a => Eq (HSMul.hSMul σ a) a) y)
          f : AlgEquiv k K K
          hf : Membership.mem (HSMul.hSMul σ ↑(FiniteGaloisIntermediateField.adjoin k (S …
          g : AlgEquiv k K K
          hg : Membership.mem (↑(FiniteGaloisIntermediateField.adjoin k (Singleton.singl …
          eq : Eq (HSMul.hSMul σ g) f
          ⊢ Not (Membership.mem L.fixingSubgroup (HSMul.hSMul σ g))
        -/
        apply (mem_fixingSubgroup_iff (K ≃ₐ[k] K)).not.mpr
        /-
          case h.left.intro.intro
          k : Type u_1
          K : Type u_2
          inst✝³ : Field k
          inst✝² : Field K
          inst✝¹ : Algebra k K
          L : IntermediateField k K
          inst✝ : IsGalois k K
          σ : AlgEquiv k K K
          h : Membership.mem (HasCompl.compl ↑L.fixingSubgroup) σ
          y : K
          yL : Membership.mem (↑L) y
          ne : Not (Membership.mem (fun a => Eq (HSMul.hSMul σ a) a) y)
          f : AlgEquiv k K K
          hf : Membership.mem (HSMul.hSMul σ ↑(FiniteGaloisIntermediateField.adjoin k (S …
          g : AlgEquiv k K K
          hg : Membership.mem (↑(FiniteGaloisIntermediateField.adjoin k (Singleton.singl …
          eq : Eq (HSMul.hSMul σ g) f
          ⊢ Not (∀ (y : K), Membership.mem (↑L) y → Eq (HSMul.hSMul (HSMul.hSMul σ g) y) …
        -/
        push_neg
        /-
          case h.left.intro.intro
          k : Type u_1
          K : Type u_2
          inst✝³ : Field k
          inst✝² : Field K
          inst✝¹ : Algebra k K
          L : IntermediateField k K
          inst✝ : IsGalois k K
          σ : AlgEquiv k K K
          h : Membership.mem (HasCompl.compl ↑L.fixingSubgroup) σ
          y : K
          yL : Membership.mem (↑L) y
          ne : Not (Membership.mem (fun a => Eq (HSMul.hSMul σ a) a) y)
          f : AlgEquiv k K K
          hf : Membership.mem (HSMul.hSMul σ ↑(FiniteGaloisIntermediateField.adjoin k (S …
          g : AlgEquiv k K K
          hg : Membership.mem (↑(FiniteGaloisIntermediateField.adjoin k (Singleton.singl …
          eq : Eq (HSMul.hSMul σ g) f
          ⊢ Exists fun y => And (Membership.mem (↑L) y) (Ne (HSMul.hSMul (HSMul.hSMul σ  …
        -/
        use y
        /-
          case h
          k : Type u_1
          K : Type u_2
          inst✝³ : Field k
          inst✝² : Field K
          inst✝¹ : Algebra k K
          L : IntermediateField k K
          inst✝ : IsGalois k K
          σ : AlgEquiv k K K
          h : Membership.mem (HasCompl.compl ↑L.fixingSubgroup) σ
          y : K
          yL : Membership.mem (↑L) y
          ne : Not (Membership.mem (fun a => Eq (HSMul.hSMul σ a) a) y)
          f : AlgEquiv k K K
          hf : Membership.mem (HSMul.hSMul σ ↑(FiniteGaloisIntermediateField.adjoin k (S …
          g : AlgEquiv k K K
          hg : Membership.mem (↑(FiniteGaloisIntermediateField.adjoin k (Singleton.singl …
          eq : Eq (HSMul.hSMul σ g) f
          ⊢ And (Membership.mem (↑L) y) (Ne (HSMul.hSMul (HSMul.hSMul σ g) y) y)
        -/
        simp only [yL, smul_eq_mul, AlgEquiv.smul_def, AlgEquiv.mul_apply, ne_eq, true_and]
        have : g y = y := (mem_fixingSubgroup_iff (K ≃ₐ[k] K)).mp hg y <|
          adjoin_simple_le_iff.mp le_rfl
        /-
          case h
          k : Type u_1
          K : Type u_2
          inst✝³ : Field k
          inst✝² : Field K
          inst✝¹ : Algebra k K
          L : IntermediateField k K
          inst✝ : IsGalois k K
          σ : AlgEquiv k K K
          h : Membership.mem (HasCompl.compl ↑L.fixingSubgroup) σ
          y : K
          yL : Membership.mem (↑L) y
          ne : Not (Membership.mem (fun a => Eq (HSMul.hSMul σ a) a) y)
          f : AlgEquiv k K K
          hf : Membership.mem (HSMul.hSMul σ ↑(FiniteGaloisIntermediateField.adjoin k (S …
          g : AlgEquiv k K K
          hg : Membership.mem (↑(FiniteGaloisIntermediateField.adjoin k (Singleton.singl …
          eq : Eq (HSMul.hSMul σ g) f
          this : Eq (g y) y
          ⊢ Not (Eq (σ (g y)) y)
        -/
        simpa only [this, ne_eq, AlgEquiv.smul_def] using ne
        /-
          🎉 no goals
        -/
        /-
          case h.right
          k : Type u_1
          K : Type u_2
          inst✝³ : Field k
          inst✝² : Field K
          inst✝¹ : Algebra k K
          L : IntermediateField k K
          inst✝ : IsGalois k K
          σ : AlgEquiv k K K
          h : Membership.mem (HasCompl.compl ↑L.fixingSubgroup) σ
          y : K
          yL : Membership.mem (↑L) y
          ne : Not (Membership.mem (fun a => Eq (HSMul.hSMul σ a) a) y)
          ⊢ And (IsOpen (HSMul.hSMul σ ↑(FiniteGaloisIntermediateField.adjoin k (Singlet …
        -/
      · simp only [(IntermediateField.fixingSubgroup_isOpen (adjoin k {y}).1).smul σ, true_and]
        /-
          case h.right
          k : Type u_1
          K : Type u_2
          inst✝³ : Field k
          inst✝² : Field K
          inst✝¹ : Algebra k K
          L : IntermediateField k K
          inst✝ : IsGalois k K
          σ : AlgEquiv k K K
          h : Membership.mem (HasCompl.compl ↑L.fixingSubgroup) σ
          y : K
          yL : Membership.mem (↑L) y
          ne : Not (Membership.mem (fun a => Eq (HSMul.hSMul σ a) a) y)
          ⊢ Membership.mem (HSMul.hSMul σ ↑(FiniteGaloisIntermediateField.adjoin k (Sing …
        -/
        use 1
        /-
          case h
          k : Type u_1
          K : Type u_2
          inst✝³ : Field k
          inst✝² : Field K
          inst✝¹ : Algebra k K
          L : IntermediateField k K
          inst✝ : IsGalois k K
          σ : AlgEquiv k K K
          h : Membership.mem (HasCompl.compl ↑L.fixingSubgroup) σ
          y : K
          yL : Membership.mem (↑L) y
          ne : Not (Membership.mem (fun a => Eq (HSMul.hSMul σ a) a) y)
          ⊢ And (Membership.mem (↑(FiniteGaloisIntermediateField.adjoin k (Singleton.sin …
        -/
        simp only [SetLike.mem_coe, smul_eq_mul, mul_one, and_true, Subgroup.one_mem]
        /-
          🎉 no goals
        -/


lemma fixedField_fixingSubgroup (L : IntermediateField k K) [IsGalois k K] :
    IntermediateField.fixedField L.fixingSubgroup = L := by
  /-
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    L : IntermediateField k K
    inst✝ : IsGalois k K
    ⊢ Eq (IntermediateField.fixedField L.fixingSubgroup) L
  -/
  apply le_antisymm
    /-
      case a
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      L : IntermediateField k K
      inst✝ : IsGalois k K
      ⊢ LE.le (IntermediateField.fixedField L.fixingSubgroup) L
    -/
  · intro x hx
    /-
      case a
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      L : IntermediateField k K
      inst✝ : IsGalois k K
      x : K
      hx : Membership.mem (IntermediateField.fixedField L.fixingSubgroup) x
      ⊢ Membership.mem L x
    -/
    rw [IntermediateField.mem_fixedField_iff] at hx
    /-
      case a
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      L : IntermediateField k K
      inst✝ : IsGalois k K
      x : K
      hx : ∀ (f : AlgEquiv k K K), Membership.mem L.fixingSubgroup f → Eq (f x) x
      ⊢ Membership.mem L x
    -/
    have mem : x ∈ (adjoin L {x}).1 := subset_adjoin _ _ rfl
    have : IntermediateField.fixedField (⊤ : Subgroup ((adjoin L {x}) ≃ₐ[L] (adjoin L {x}))) = ⊥ :=
      (IsGalois.tfae.out 0 1).mp (by infer_instance)
    have : ⟨x, mem⟩ ∈ (⊥ : IntermediateField L (adjoin L {x})) := by
      rw [← this, IntermediateField.mem_fixedField_iff]
      intro f _
      rcases restrictNormalHom_surjective K f with ⟨σ,hσ⟩
      apply Subtype.val_injective
      rw [← hσ, restrictNormalHom_apply (adjoin L {x}).1 σ ⟨x, mem⟩]
      have := hx ((IntermediateField.fixingSubgroupEquiv L).symm σ)
      simpa only [SetLike.coe_mem, true_implies]
    /-
      case a
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      L : IntermediateField k K
      inst✝ : IsGalois k K
      x : K
      hx : ∀ (f : AlgEquiv k K K), Membership.mem L.fixingSubgroup f → Eq (f x) x
      mem : Membership.mem (FiniteGaloisIntermediateField.adjoin (Subtype fun x => M …
      this✝ : Eq (IntermediateField.fixedField Top.top) Bot.bot
      this : Membership.mem Bot.bot ⟨x, mem⟩
      ⊢ Membership.mem L x
    -/
    rcases IntermediateField.mem_bot.mp this with ⟨y, hy⟩
    /-
      case a.intro
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      L : IntermediateField k K
      inst✝ : IsGalois k K
      x : K
      hx : ∀ (f : AlgEquiv k K K), Membership.mem L.fixingSubgroup f → Eq (f x) x
      mem : Membership.mem (FiniteGaloisIntermediateField.adjoin (Subtype fun x => M …
      this✝ : Eq (IntermediateField.fixedField Top.top) Bot.bot
      this : Membership.mem Bot.bot ⟨x, mem⟩
      y : Subtype fun x => Membership.mem L x
      hy : Eq ((algebraMap (Subtype fun x => Membership.mem L x) (Subtype fun x_1 => …
      ⊢ Membership.mem L x
    -/
    obtain ⟨rfl⟩ : y = x := congrArg Subtype.val hy
    /-
      case a.intro.refl
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      L : IntermediateField k K
      inst✝ : IsGalois k K
      y : Subtype fun x => Membership.mem L x
      hx : ∀ (f : AlgEquiv k K K), Membership.mem L.fixingSubgroup f → Eq (f ↑y) ↑y
      mem : Membership.mem (FiniteGaloisIntermediateField.adjoin (Subtype fun x => M …
      this✝ : Eq (IntermediateField.fixedField Top.top) Bot.bot
      this : Membership.mem Bot.bot ⟨↑y, mem⟩
      hy : Eq ((algebraMap (Subtype fun x => Membership.mem L x) (Subtype fun x => M …
      ⊢ Membership.mem L ↑y
    -/
    exact y.2
    /-
      🎉 no goals
    -/
    /-
      case a
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      L : IntermediateField k K
      inst✝ : IsGalois k K
      ⊢ LE.le L (IntermediateField.fixedField L.fixingSubgroup)
    -/
  · exact (IntermediateField.le_iff_le L.fixingSubgroup L).mpr le_rfl
    /-
      🎉 no goals
    -/


lemma fixedField_bot [IsGalois k K] :
    IntermediateField.fixedField (⊤ : Subgroup (K ≃ₐ[k] K)) = ⊥ := by
  /-
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    ⊢ Eq (IntermediateField.fixedField Top.top) Bot.bot
  -/
  rw [← IntermediateField.fixingSubgroup_bot, fixedField_fixingSubgroup]
  /-
    🎉 no goals
  -/


open IntermediateField in
/--For a subgroup `H` of `Gal(K/k)`, the fixed field of the image of `H` under the restriction to
a normal intermediate field `E` is equal to the fixed field of `H` in `K` intersecting with `E`.-/
lemma restrict_fixedField (H : Subgroup (K ≃ₐ[k] K)) (L : IntermediateField k K) [Normal k L] :
    fixedField H ⊓ L = lift (fixedField (Subgroup.map (restrictNormalHom L) H)) := by
  /-
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    H : Subgroup (AlgEquiv k K K)
    L : IntermediateField k K
    inst✝ : Normal k (Subtype fun x => Membership.mem L x)
    ⊢ Eq (Min.min (IntermediateField.fixedField H) L) (IntermediateField.lift (Int …
  -/
  apply SetLike.ext'
  /-
    case h
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    H : Subgroup (AlgEquiv k K K)
    L : IntermediateField k K
    inst✝ : Normal k (Subtype fun x => Membership.mem L x)
    ⊢ Eq ↑(Min.min (IntermediateField.fixedField H) L) ↑(IntermediateField.lift (I …
  -/
  ext x
  /-
    case h.h
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    H : Subgroup (AlgEquiv k K K)
    L : IntermediateField k K
    inst✝ : Normal k (Subtype fun x => Membership.mem L x)
    x : K
    ⊢ Iff (Membership.mem (↑(Min.min (IntermediateField.fixedField H) L)) x) (Memb …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case h.h.refine_1
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      H : Subgroup (AlgEquiv k K K)
      L : IntermediateField k K
      inst✝ : Normal k (Subtype fun x => Membership.mem L x)
      x : K
      h : Membership.mem (↑(Min.min (IntermediateField.fixedField H) L)) x
      ⊢ Membership.mem (↑(IntermediateField.lift (IntermediateField.fixedField (Subg …
    -/
  · have xL := h.out.2
    /-
      case h.h.refine_1
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      H : Subgroup (AlgEquiv k K K)
      L : IntermediateField k K
      inst✝ : Normal k (Subtype fun x => Membership.mem L x)
      x : K
      h : Membership.mem (↑(Min.min (IntermediateField.fixedField H) L)) x
      xL : Membership.mem ((fun x => ↑x) L) x
      ⊢ Membership.mem (↑(IntermediateField.lift (IntermediateField.fixedField (Subg …
    -/
    apply (mem_lift (⟨x, xL⟩ : L)).mpr
    simp only [mem_fixedField_iff, Subgroup.mem_map, forall_exists_index, and_imp,
      forall_apply_eq_imp_iff₂]
    /-
      case h.h.refine_1
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      H : Subgroup (AlgEquiv k K K)
      L : IntermediateField k K
      inst✝ : Normal k (Subtype fun x => Membership.mem L x)
      x : K
      h : Membership.mem (↑(Min.min (IntermediateField.fixedField H) L)) x
      xL : Membership.mem ((fun x => ↑x) L) x
      ⊢ ∀ (a : AlgEquiv k K K), Membership.mem H a → Eq (((AlgEquiv.restrictNormalHo …
    -/
    intro σ hσ
    /-
      case h.h.refine_1
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      H : Subgroup (AlgEquiv k K K)
      L : IntermediateField k K
      inst✝ : Normal k (Subtype fun x => Membership.mem L x)
      x : K
      h : Membership.mem (↑(Min.min (IntermediateField.fixedField H) L)) x
      xL : Membership.mem ((fun x => ↑x) L) x
      σ : AlgEquiv k K K
      hσ : Membership.mem H σ
      ⊢ Eq (((AlgEquiv.restrictNormalHom (Subtype fun x => Membership.mem L x)) σ) ⟨ …
    -/
    apply Subtype.val_injective
    /-
      case h.h.refine_1.a
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      H : Subgroup (AlgEquiv k K K)
      L : IntermediateField k K
      inst✝ : Normal k (Subtype fun x => Membership.mem L x)
      x : K
      h : Membership.mem (↑(Min.min (IntermediateField.fixedField H) L)) x
      xL : Membership.mem ((fun x => ↑x) L) x
      σ : AlgEquiv k K K
      hσ : Membership.mem H σ
      ⊢ Eq ↑(((AlgEquiv.restrictNormalHom (Subtype fun x => Membership.mem L x)) σ)  …
    -/
    dsimp only
    /-
      case h.h.refine_1.a
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      H : Subgroup (AlgEquiv k K K)
      L : IntermediateField k K
      inst✝ : Normal k (Subtype fun x => Membership.mem L x)
      x : K
      h : Membership.mem (↑(Min.min (IntermediateField.fixedField H) L)) x
      xL : Membership.mem ((fun x => ↑x) L) x
      σ : AlgEquiv k K K
      hσ : Membership.mem H σ
      ⊢ Eq (↑(((AlgEquiv.restrictNormalHom (Subtype fun x => Membership.mem L x)) σ) …
    -/
    nth_rw 2 [← (h.out.1 ⟨σ, hσ⟩)]
    /-
      case h.h.refine_1.a
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      H : Subgroup (AlgEquiv k K K)
      L : IntermediateField k K
      inst✝ : Normal k (Subtype fun x => Membership.mem L x)
      x : K
      h : Membership.mem (↑(Min.min (IntermediateField.fixedField H) L)) x
      xL : Membership.mem ((fun x => ↑x) L) x
      σ : AlgEquiv k K K
      hσ : Membership.mem H σ
      ⊢ Eq (↑(((AlgEquiv.restrictNormalHom (Subtype fun x => Membership.mem L x)) σ) …
    -/
    exact AlgEquiv.restrictNormal_commutes σ L ⟨x, xL⟩
    /-
      🎉 no goals
    -/
    /-
      case h.h.refine_2
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      H : Subgroup (AlgEquiv k K K)
      L : IntermediateField k K
      inst✝ : Normal k (Subtype fun x => Membership.mem L x)
      x : K
      h : Membership.mem (↑(IntermediateField.lift (IntermediateField.fixedField (Su …
      ⊢ Membership.mem (↑(Min.min (IntermediateField.fixedField H) L)) x
    -/
  · have xL := lift_le _ h
    /-
      case h.h.refine_2
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      H : Subgroup (AlgEquiv k K K)
      L : IntermediateField k K
      inst✝ : Normal k (Subtype fun x => Membership.mem L x)
      x : K
      h : Membership.mem (↑(IntermediateField.lift (IntermediateField.fixedField (Su …
      xL : Membership.mem L x
      ⊢ Membership.mem (↑(Min.min (IntermediateField.fixedField H) L)) x
    -/
    apply (mem_lift (⟨x,xL⟩ : L)).mp at h
    simp only [mem_fixedField_iff, Subgroup.mem_map, forall_exists_index, and_imp,
      forall_apply_eq_imp_iff₂] at h
    /-
      case h.h.refine_2
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      H : Subgroup (AlgEquiv k K K)
      L : IntermediateField k K
      inst✝ : Normal k (Subtype fun x => Membership.mem L x)
      x : K
      xL : Membership.mem L x
      h : ∀ (a : AlgEquiv k K K), Membership.mem H a → Eq (((AlgEquiv.restrictNormal …
      ⊢ Membership.mem (↑(Min.min (IntermediateField.fixedField H) L)) x
    -/
    simp only [coe_inf, Set.mem_inter_iff, SetLike.mem_coe, mem_fixedField_iff, xL, and_true]
    /-
      case h.h.refine_2
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      H : Subgroup (AlgEquiv k K K)
      L : IntermediateField k K
      inst✝ : Normal k (Subtype fun x => Membership.mem L x)
      x : K
      xL : Membership.mem L x
      h : ∀ (a : AlgEquiv k K K), Membership.mem H a → Eq (((AlgEquiv.restrictNormal …
      ⊢ ∀ (f : AlgEquiv k K K), Membership.mem H f → Eq (f x) x
    -/
    intro σ hσ
    /-
      case h.h.refine_2
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      H : Subgroup (AlgEquiv k K K)
      L : IntermediateField k K
      inst✝ : Normal k (Subtype fun x => Membership.mem L x)
      x : K
      xL : Membership.mem L x
      h : ∀ (a : AlgEquiv k K K), Membership.mem H a → Eq (((AlgEquiv.restrictNormal …
      σ : AlgEquiv k K K
      hσ : Membership.mem H σ
      ⊢ Eq (σ x) x
    -/
    have : ((restrictNormalHom L σ) ⟨x, xL⟩).1 = x := by rw [h σ hσ]
    /-
      case h.h.refine_2
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      H : Subgroup (AlgEquiv k K K)
      L : IntermediateField k K
      inst✝ : Normal k (Subtype fun x => Membership.mem L x)
      x : K
      xL : Membership.mem L x
      h : ∀ (a : AlgEquiv k K K), Membership.mem H a → Eq (((AlgEquiv.restrictNormal …
      σ : AlgEquiv k K K
      hσ : Membership.mem H σ
      this : Eq (↑(((AlgEquiv.restrictNormalHom (Subtype fun x => Membership.mem L x …
      ⊢ Eq (σ x) x
    -/
    nth_rw 2 [← this]
    /-
      case h.h.refine_2
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      H : Subgroup (AlgEquiv k K K)
      L : IntermediateField k K
      inst✝ : Normal k (Subtype fun x => Membership.mem L x)
      x : K
      xL : Membership.mem L x
      h : ∀ (a : AlgEquiv k K K), Membership.mem H a → Eq (((AlgEquiv.restrictNormal …
      σ : AlgEquiv k K K
      hσ : Membership.mem H σ
      this : Eq (↑(((AlgEquiv.restrictNormalHom (Subtype fun x => Membership.mem L x …
      ⊢ Eq (σ x) ↑(((AlgEquiv.restrictNormalHom (Subtype fun x => Membership.mem L x …
    -/
    exact (AlgEquiv.restrictNormal_commutes σ L ⟨x, xL⟩).symm
    /-
      🎉 no goals
    -/


lemma fixingSubgroup_fixedField (H : ClosedSubgroup (K ≃ₐ[k] K)) [IsGalois k K] :
    (IntermediateField.fixedField H).fixingSubgroup = H.1 := by
  apply le_antisymm _ ((IntermediateField.le_iff_le H.toSubgroup
    (IntermediateField.fixedField H.toSubgroup)).mp le_rfl)
  /-
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    H : ClosedSubgroup (AlgEquiv k K K)
    inst✝ : IsGalois k K
    ⊢ LE.le (IntermediateField.fixedField ↑H).fixingSubgroup ↑H
  -/
  intro σ hσ
  /-
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    H : ClosedSubgroup (AlgEquiv k K K)
    inst✝ : IsGalois k K
    σ : AlgEquiv k K K
    hσ : Membership.mem (IntermediateField.fixedField ↑H).fixingSubgroup σ
    ⊢ Membership.mem (↑H) σ
  -/
  by_contra h
  /-
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    H : ClosedSubgroup (AlgEquiv k K K)
    inst✝ : IsGalois k K
    σ : AlgEquiv k K K
    hσ : Membership.mem (IntermediateField.fixedField ↑H).fixingSubgroup σ
    h : Not (Membership.mem (↑H) σ)
    ⊢ False
  -/
  have nhd : H.carrierᶜ ∈ nhds σ := H.isClosed'.isOpen_compl.mem_nhds h
  /-
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    H : ClosedSubgroup (AlgEquiv k K K)
    inst✝ : IsGalois k K
    σ : AlgEquiv k K K
    hσ : Membership.mem (IntermediateField.fixedField ↑H).fixingSubgroup σ
    h : Not (Membership.mem (↑H) σ)
    nhd : Membership.mem (nhds σ) (HasCompl.compl (↑H).carrier)
    ⊢ False
  -/
  rw [GroupFilterBasis.nhds_eq (x₀ := σ) (galGroupBasis k K)] at nhd
  /-
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    H : ClosedSubgroup (AlgEquiv k K K)
    inst✝ : IsGalois k K
    σ : AlgEquiv k K K
    hσ : Membership.mem (IntermediateField.fixedField ↑H).fixingSubgroup σ
    h : Not (Membership.mem (↑H) σ)
    nhd : Membership.mem ((galGroupBasis k K).N σ) (HasCompl.compl (↑H).carrier)
    ⊢ False
  -/
  rcases nhd with ⟨b, ⟨gp, ⟨L, hL, eq'⟩, eq⟩, sub⟩
  /-
    case intro.intro.intro.intro.intro.intro
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    H : ClosedSubgroup (AlgEquiv k K K)
    inst✝ : IsGalois k K
    σ : AlgEquiv k K K
    hσ : Membership.mem (IntermediateField.fixedField ↑H).fixingSubgroup σ
    h : Not (Membership.mem (↑H) σ)
    b : Set (AlgEquiv k K K)
    sub : HasSubset.Subset b (Set.preimage (fun y => HMul.hMul σ y) (HasCompl.comp …
    gp : Subgroup (AlgEquiv k K K)
    eq : Eq ((fun g => g.carrier) gp) b
    L : IntermediateField k K
    hL : Membership.mem (finiteExts k K) L
    eq' : Eq L.fixingSubgroup gp
    ⊢ False
  -/
  rw [← eq'] at eq
  /-
    case intro.intro.intro.intro.intro.intro
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    H : ClosedSubgroup (AlgEquiv k K K)
    inst✝ : IsGalois k K
    σ : AlgEquiv k K K
    hσ : Membership.mem (IntermediateField.fixedField ↑H).fixingSubgroup σ
    h : Not (Membership.mem (↑H) σ)
    b : Set (AlgEquiv k K K)
    sub : HasSubset.Subset b (Set.preimage (fun y => HMul.hMul σ y) (HasCompl.comp …
    gp : Subgroup (AlgEquiv k K K)
    L : IntermediateField k K
    eq : Eq ((fun g => g.carrier) L.fixingSubgroup) b
    hL : Membership.mem (finiteExts k K) L
    eq' : Eq L.fixingSubgroup gp
    ⊢ False
  -/
  have := hL.out
  let L' : FiniteGaloisIntermediateField k K := {
    normalClosure k L K with
    finiteDimensional := normalClosure.is_finiteDimensional k L K
    isGalois := IsGalois.normalClosure k L K }
  have compl : σ • L'.1.fixingSubgroup.carrier ⊆ H.carrierᶜ := by
    rintro φ ⟨τ, hτ, muleq⟩
    have sub' : σ • b ⊆ H.carrierᶜ := Set.smul_set_subset_iff.mpr sub
    apply sub'
    simp only [← muleq, ← eq]
    apply Set.smul_mem_smul_set
    exact (IntermediateField.fixingSubgroup.antimono (IntermediateField.le_normalClosure L) hτ)
  have fix : ∀ x ∈ IntermediateField.fixedField H.toSubgroup ⊓ ↑L', σ x = x :=
    fun x hx ↦ ((mem_fixingSubgroup_iff (K ≃ₐ[k] K)).mp hσ) x hx.1
  /-
    case intro.intro.intro.intro.intro.intro
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    H : ClosedSubgroup (AlgEquiv k K K)
    inst✝ : IsGalois k K
    σ : AlgEquiv k K K
    hσ : Membership.mem (IntermediateField.fixedField ↑H).fixingSubgroup σ
    h : Not (Membership.mem (↑H) σ)
    b : Set (AlgEquiv k K K)
    sub : HasSubset.Subset b (Set.preimage (fun y => HMul.hMul σ y) (HasCompl.comp …
    gp : Subgroup (AlgEquiv k K K)
    L : IntermediateField k K
    eq : Eq ((fun g => g.carrier) L.fixingSubgroup) b
    hL : Membership.mem (finiteExts k K) L
    eq' : Eq L.fixingSubgroup gp
    this : FiniteDimensional k (Subtype fun x => Membership.mem L x)
    L' : FiniteGaloisIntermediateField k K :=
      let __src := normalClosure k (Subtype fun x => Membership.mem L x) K;
      FiniteGaloisIntermediateField.mk __src
    compl : HasSubset.Subset (HSMul.hSMul σ L'.fixingSubgroup.carrier) (HasCompl.c …
    fix : ∀ (x : K), Membership.mem (Min.min (IntermediateField.fixedField ↑H) L'. …
    ⊢ False
  -/
  rw [restrict_fixedField H.1 L'.1] at fix
  have : (restrictNormalHom L') σ ∈ (Subgroup.map (restrictNormalHom L') H.1) := by
    rw [← IntermediateField.fixingSubgroup_fixedField (Subgroup.map (restrictNormalHom L') H.1)]
    apply (mem_fixingSubgroup_iff (L' ≃ₐ[k] L')).mpr
    intro y hy
    apply Subtype.val_injective
    simp only [AlgEquiv.smul_def, restrictNormalHom_apply L'.1 σ y,
      fix y.1 ((IntermediateField.mem_lift y).mpr hy)]
  /-
    case intro.intro.intro.intro.intro.intro
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    H : ClosedSubgroup (AlgEquiv k K K)
    inst✝ : IsGalois k K
    σ : AlgEquiv k K K
    hσ : Membership.mem (IntermediateField.fixedField ↑H).fixingSubgroup σ
    h : Not (Membership.mem (↑H) σ)
    b : Set (AlgEquiv k K K)
    sub : HasSubset.Subset b (Set.preimage (fun y => HMul.hMul σ y) (HasCompl.comp …
    gp : Subgroup (AlgEquiv k K K)
    L : IntermediateField k K
    eq : Eq ((fun g => g.carrier) L.fixingSubgroup) b
    hL : Membership.mem (finiteExts k K) L
    eq' : Eq L.fixingSubgroup gp
    this✝ : FiniteDimensional k (Subtype fun x => Membership.mem L x)
    L' : FiniteGaloisIntermediateField k K :=
      let __src := normalClosure k (Subtype fun x => Membership.mem L x) K;
      FiniteGaloisIntermediateField.mk __src
    compl : HasSubset.Subset (HSMul.hSMul σ L'.fixingSubgroup.carrier) (HasCompl.c …
    fix : ∀ (x : K), Membership.mem (IntermediateField.lift (IntermediateField.fix …
    this : Membership.mem (Subgroup.map (AlgEquiv.restrictNormalHom (Subtype fun x …
    ⊢ False
  -/
  rcases this with ⟨h, mem, eq⟩
  have : h ∈ σ • L'.1.fixingSubgroup.carrier := by
    use σ⁻¹ * h
    simp only [Subsemigroup.mem_carrier, Submonoid.mem_toSubsemigroup, Subgroup.mem_toSubmonoid,
      smul_eq_mul, mul_inv_cancel_left, and_true]
    apply (mem_fixingSubgroup_iff (K ≃ₐ[k] K)).mpr
    intro y hy
    simp only [AlgEquiv.smul_def, AlgEquiv.mul_apply]
    have : ((restrictNormalHom L') h ⟨y,hy⟩).1 = ((restrictNormalHom L') σ ⟨y,hy⟩).1 := by rw [eq]
    rw [restrictNormalHom_apply L'.1 h ⟨y, hy⟩, restrictNormalHom_apply L'.1 σ ⟨y, hy⟩] at this
    simp only [this, ← AlgEquiv.mul_apply, inv_mul_cancel, one_apply]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    H : ClosedSubgroup (AlgEquiv k K K)
    inst✝ : IsGalois k K
    σ : AlgEquiv k K K
    hσ : Membership.mem (IntermediateField.fixedField ↑H).fixingSubgroup σ
    h✝ : Not (Membership.mem (↑H) σ)
    b : Set (AlgEquiv k K K)
    sub : HasSubset.Subset b (Set.preimage (fun y => HMul.hMul σ y) (HasCompl.comp …
    gp : Subgroup (AlgEquiv k K K)
    L : IntermediateField k K
    eq✝ : Eq ((fun g => g.carrier) L.fixingSubgroup) b
    hL : Membership.mem (finiteExts k K) L
    eq' : Eq L.fixingSubgroup gp
    this✝ : FiniteDimensional k (Subtype fun x => Membership.mem L x)
    L' : FiniteGaloisIntermediateField k K :=
      let __src := normalClosure k (Subtype fun x => Membership.mem L x) K;
      FiniteGaloisIntermediateField.mk __src
    compl : HasSubset.Subset (HSMul.hSMul σ L'.fixingSubgroup.carrier) (HasCompl.c …
    fix : ∀ (x : K), Membership.mem (IntermediateField.lift (IntermediateField.fix …
    h : AlgEquiv k K K
    mem : Membership.mem (↑↑H) h
    eq : Eq ((AlgEquiv.restrictNormalHom (Subtype fun x => Membership.mem L'.toInt …
    this : Membership.mem (HSMul.hSMul σ L'.fixingSubgroup.carrier) h
    ⊢ False
  -/
  absurd compl
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    H : ClosedSubgroup (AlgEquiv k K K)
    inst✝ : IsGalois k K
    σ : AlgEquiv k K K
    hσ : Membership.mem (IntermediateField.fixedField ↑H).fixingSubgroup σ
    h✝ : Not (Membership.mem (↑H) σ)
    b : Set (AlgEquiv k K K)
    sub : HasSubset.Subset b (Set.preimage (fun y => HMul.hMul σ y) (HasCompl.comp …
    gp : Subgroup (AlgEquiv k K K)
    L : IntermediateField k K
    eq✝ : Eq ((fun g => g.carrier) L.fixingSubgroup) b
    hL : Membership.mem (finiteExts k K) L
    eq' : Eq L.fixingSubgroup gp
    this✝ : FiniteDimensional k (Subtype fun x => Membership.mem L x)
    L' : FiniteGaloisIntermediateField k K :=
      let __src := normalClosure k (Subtype fun x => Membership.mem L x) K;
      FiniteGaloisIntermediateField.mk __src
    compl : HasSubset.Subset (HSMul.hSMul σ L'.fixingSubgroup.carrier) (HasCompl.c …
    fix : ∀ (x : K), Membership.mem (IntermediateField.lift (IntermediateField.fix …
    h : AlgEquiv k K K
    mem : Membership.mem (↑↑H) h
    eq : Eq ((AlgEquiv.restrictNormalHom (Subtype fun x => Membership.mem L'.toInt …
    this : Membership.mem (HSMul.hSMul σ L'.fixingSubgroup.carrier) h
    ⊢ Not (HasSubset.Subset (HSMul.hSMul σ L'.fixingSubgroup.carrier) (HasCompl.co …
  -/
  apply Set.not_subset.mpr
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    H : ClosedSubgroup (AlgEquiv k K K)
    inst✝ : IsGalois k K
    σ : AlgEquiv k K K
    hσ : Membership.mem (IntermediateField.fixedField ↑H).fixingSubgroup σ
    h✝ : Not (Membership.mem (↑H) σ)
    b : Set (AlgEquiv k K K)
    sub : HasSubset.Subset b (Set.preimage (fun y => HMul.hMul σ y) (HasCompl.comp …
    gp : Subgroup (AlgEquiv k K K)
    L : IntermediateField k K
    eq✝ : Eq ((fun g => g.carrier) L.fixingSubgroup) b
    hL : Membership.mem (finiteExts k K) L
    eq' : Eq L.fixingSubgroup gp
    this✝ : FiniteDimensional k (Subtype fun x => Membership.mem L x)
    L' : FiniteGaloisIntermediateField k K :=
      let __src := normalClosure k (Subtype fun x => Membership.mem L x) K;
      FiniteGaloisIntermediateField.mk __src
    compl : HasSubset.Subset (HSMul.hSMul σ L'.fixingSubgroup.carrier) (HasCompl.c …
    fix : ∀ (x : K), Membership.mem (IntermediateField.lift (IntermediateField.fix …
    h : AlgEquiv k K K
    mem : Membership.mem (↑↑H) h
    eq : Eq ((AlgEquiv.restrictNormalHom (Subtype fun x => Membership.mem L'.toInt …
    this : Membership.mem (HSMul.hSMul σ L'.fixingSubgroup.carrier) h
    ⊢ Exists fun a => And (Membership.mem (HSMul.hSMul σ L'.fixingSubgroup.carrier …
  -/
  use h
  simpa only [this, Set.mem_compl_iff, Subsemigroup.mem_carrier, Submonoid.mem_toSubsemigroup,
    Subgroup.mem_toSubmonoid, not_not, true_and] using mem


/-- The Galois correspondence from intermediate fields to closed subgroups. -/
def IntermediateFieldEquivClosedSubgroup [IsGalois k K] :
    IntermediateField k K ≃o (ClosedSubgroup (K ≃ₐ[k] K))ᵒᵈ where
  toFun := fun L =>
    { L.fixingSubgroup with
      isClosed' := fixingSubgroup_isClosed L }
  invFun := fun H => IntermediateField.fixedField H.1
  left_inv := fun L => fixedField_fixingSubgroup L
  right_inv := by
    /-
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      inst✝ : IsGalois k K
      ⊢ Function.RightInverse (fun H => IntermediateField.fixedField ↑H) fun L =>
          let __src := L.fixingSubgroup;
          { toSubgroup := __src, isClosed' := ⋯ }
    -/
    intro H
    /-
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      inst✝ : IsGalois k K
      H : OrderDual (ClosedSubgroup (AlgEquiv k K K))
      ⊢ Eq
          ((fun L =>
              let __src := L.fixingSubgroup;
              { toSubgroup := __src, isClosed' := ⋯ })
            ((fun H => IntermediateField.fixedField ↑H) H))
          H
    -/
    simp_rw [fixingSubgroup_fixedField H]
    /-
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      inst✝ : IsGalois k K
      H : OrderDual (ClosedSubgroup (AlgEquiv k K K))
      ⊢ Eq { toSubgroup := ↑H, isClosed' := ⋯ } H
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_rel_iff' := by
    /-
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      inst✝ : IsGalois k K
      ⊢ ∀ {a b : IntermediateField k K},
          Iff
            (LE.le
              ({
                  toFun := fun L =>
                    let __src := L.fixingSubgroup;
                    { toSubgroup := __src, isClosed' := ⋯ },
                  invFun := fun H => IntermediateField.fixedField ↑H, left_inv := ⋯, …
                a)
              ({
                  toFun := fun L =>
                    let __src := L.fixingSubgroup;
                    { toSubgroup := __src, isClosed' := ⋯ },
                  invFun := fun H => IntermediateField.fixedField ↑H, left_inv := ⋯, …
                b))
            (LE.le a b)
    -/
    intro L₁ L₂
    /-
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      inst✝ : IsGalois k K
      L₁ L₂ : IntermediateField k K
      ⊢ Iff
          (LE.le
            ({
                toFun := fun L =>
                  let __src := L.fixingSubgroup;
                  { toSubgroup := __src, isClosed' := ⋯ },
                invFun := fun H => IntermediateField.fixedField ↑H, left_inv := ⋯, r …
              L₁)
            ({
                toFun := fun L =>
                  let __src := L.fixingSubgroup;
                  { toSubgroup := __src, isClosed' := ⋯ },
                invFun := fun H => IntermediateField.fixedField ↑H, left_inv := ⋯, r …
              L₂))
          (LE.le L₁ L₂)
    -/
    show L₁.fixingSubgroup ≥ L₂.fixingSubgroup ↔ L₁ ≤ L₂
    /-
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      inst✝ : IsGalois k K
      L₁ L₂ : IntermediateField k K
      ⊢ Iff (GE.ge L₁.fixingSubgroup L₂.fixingSubgroup) (LE.le L₁ L₂)
    -/
    rw [← fixedField_fixingSubgroup L₂, IntermediateField.le_iff_le, fixedField_fixingSubgroup L₂]
    /-
      🎉 no goals
    -/


/-- The Galois correspondence as a `GaloisInsertion` -/
def GaloisInsertionIntermediateFieldClosedSubgroup [IsGalois k K] :
    GaloisInsertion (OrderDual.toDual ∘ fun (E : IntermediateField k K) ↦
      (⟨E.fixingSubgroup, fixingSubgroup_isClosed E⟩ : ClosedSubgroup (K ≃ₐ[k] K)))
      ((fun (H : ClosedSubgroup (K ≃ₐ[k] K)) ↦ IntermediateField.fixedField H) ∘
        OrderDual.toDual) :=
  OrderIso.toGaloisInsertion IntermediateFieldEquivClosedSubgroup


/-- The Galois correspondence as a `GaloisCoinsertion` -/
def GaloisCoinsertionIntermediateFieldSubgroup [IsGalois k K] :
    GaloisCoinsertion (OrderDual.toDual ∘ fun (E : IntermediateField k K) ↦ E.fixingSubgroup)
      ((fun (H : Subgroup (K ≃ₐ[k] K)) ↦ IntermediateField.fixedField H) ∘
        OrderDual.toDual) where
  choice H _ := IntermediateField.fixedField H
  gc E H := (IntermediateField.le_iff_le H E).symm
  u_l_le K := le_of_eq (fixedField_fixingSubgroup K)
  choice_eq _ _ := rfl


theorem isOpen_iff_finite (L : IntermediateField k K) [IsGalois k K] :
    IsOpen (IntermediateFieldEquivClosedSubgroup L).carrier ↔
    (FiniteDimensional k L) := by
  /-
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    L : IntermediateField k K
    inst✝ : IsGalois k K
    ⊢ Iff (IsOpen (↑(InfiniteGalois.IntermediateFieldEquivClosedSubgroup L)).carri …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ IntermediateField.fixingSubgroup_isOpen L⟩
  have : (IntermediateFieldEquivClosedSubgroup.toFun L).carrier ∈ nhds 1 :=
    IsOpen.mem_nhds h (congrFun rfl)
  /-
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    L : IntermediateField k K
    inst✝ : IsGalois k K
    h : IsOpen (↑(InfiniteGalois.IntermediateFieldEquivClosedSubgroup L)).carrier
    this : Membership.mem (nhds 1) (↑(InfiniteGalois.IntermediateFieldEquivClosedS …
    ⊢ FiniteDimensional k (Subtype fun x => Membership.mem L x)
  -/
  rw [GroupFilterBasis.nhds_one_eq] at this
  /-
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    L : IntermediateField k K
    inst✝ : IsGalois k K
    h : IsOpen (↑(InfiniteGalois.IntermediateFieldEquivClosedSubgroup L)).carrier
    this : Membership.mem GroupFilterBasis.toFilterBasis.filter (↑(InfiniteGalois. …
    ⊢ FiniteDimensional k (Subtype fun x => Membership.mem L x)
  -/
  rcases this with ⟨S, ⟨gp, ⟨M, hM, eq'⟩, eq⟩, sub⟩
  /-
    case intro.intro.intro.intro.intro.intro
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    L : IntermediateField k K
    inst✝ : IsGalois k K
    h : IsOpen (↑(InfiniteGalois.IntermediateFieldEquivClosedSubgroup L)).carrier
    S : Set (AlgEquiv k K K)
    sub : HasSubset.Subset S (↑(InfiniteGalois.IntermediateFieldEquivClosedSubgrou …
    gp : Subgroup (AlgEquiv k K K)
    eq : Eq ((fun g => g.carrier) gp) S
    M : IntermediateField k K
    hM : Membership.mem (finiteExts k K) M
    eq' : Eq M.fixingSubgroup gp
    ⊢ FiniteDimensional k (Subtype fun x => Membership.mem L x)
  -/
  rw [← eq, ← eq'] at sub
  /-
    case intro.intro.intro.intro.intro.intro
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    L : IntermediateField k K
    inst✝ : IsGalois k K
    h : IsOpen (↑(InfiniteGalois.IntermediateFieldEquivClosedSubgroup L)).carrier
    S : Set (AlgEquiv k K K)
    gp : Subgroup (AlgEquiv k K K)
    eq : Eq ((fun g => g.carrier) gp) S
    M : IntermediateField k K
    sub : HasSubset.Subset ((fun g => g.carrier) M.fixingSubgroup) (↑(InfiniteGalo …
    hM : Membership.mem (finiteExts k K) M
    eq' : Eq M.fixingSubgroup gp
    ⊢ FiniteDimensional k (Subtype fun x => Membership.mem L x)
  -/
  have := hM.out
  let L' : FiniteGaloisIntermediateField k K := {
    normalClosure k M K with
    finiteDimensional := normalClosure.is_finiteDimensional k M K
    isGalois := IsGalois.normalClosure k M K }
  have : L ≤ L'.1 := by
    apply LE.le.trans _ (IntermediateField.le_normalClosure M)
    rw [←  fixedField_fixingSubgroup M, IntermediateField.le_iff_le]
    exact sub
  /-
    case intro.intro.intro.intro.intro.intro
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    L : IntermediateField k K
    inst✝ : IsGalois k K
    h : IsOpen (↑(InfiniteGalois.IntermediateFieldEquivClosedSubgroup L)).carrier
    S : Set (AlgEquiv k K K)
    gp : Subgroup (AlgEquiv k K K)
    eq : Eq ((fun g => g.carrier) gp) S
    M : IntermediateField k K
    sub : HasSubset.Subset ((fun g => g.carrier) M.fixingSubgroup) (↑(InfiniteGalo …
    hM : Membership.mem (finiteExts k K) M
    eq' : Eq M.fixingSubgroup gp
    this✝ : FiniteDimensional k (Subtype fun x => Membership.mem M x)
    L' : FiniteGaloisIntermediateField k K :=
      let __src := normalClosure k (Subtype fun x => Membership.mem M x) K;
      FiniteGaloisIntermediateField.mk __src
    this : LE.le L L'.toIntermediateField
    ⊢ FiniteDimensional k (Subtype fun x => Membership.mem L x)
  -/
  let _ : Algebra L L'.1 := RingHom.toAlgebra (IntermediateField.inclusion this)
  /-
    case intro.intro.intro.intro.intro.intro
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    L : IntermediateField k K
    inst✝ : IsGalois k K
    h : IsOpen (↑(InfiniteGalois.IntermediateFieldEquivClosedSubgroup L)).carrier
    S : Set (AlgEquiv k K K)
    gp : Subgroup (AlgEquiv k K K)
    eq : Eq ((fun g => g.carrier) gp) S
    M : IntermediateField k K
    sub : HasSubset.Subset ((fun g => g.carrier) M.fixingSubgroup) (↑(InfiniteGalo …
    hM : Membership.mem (finiteExts k K) M
    eq' : Eq M.fixingSubgroup gp
    this✝ : FiniteDimensional k (Subtype fun x => Membership.mem M x)
    L' : FiniteGaloisIntermediateField k K :=
      let __src := normalClosure k (Subtype fun x => Membership.mem M x) K;
      FiniteGaloisIntermediateField.mk __src
    this : LE.le L L'.toIntermediateField
    x✝ : Algebra (Subtype fun x => Membership.mem L x) (Subtype fun x => Membershi …
    ⊢ FiniteDimensional k (Subtype fun x => Membership.mem L x)
  -/
  exact FiniteDimensional.left k L L'.1
  /-
    🎉 no goals
  -/


theorem normal_iff_isGalois (L : IntermediateField k K) [IsGalois k K] :
    Subgroup.Normal (IntermediateFieldEquivClosedSubgroup L).1 ↔
    IsGalois k L := by
  /-
    k : Type u_1
    K : Type u_2
    inst✝³ : Field k
    inst✝² : Field K
    inst✝¹ : Algebra k K
    L : IntermediateField k K
    inst✝ : IsGalois k K
    ⊢ Iff (↑(InfiniteGalois.IntermediateFieldEquivClosedSubgroup L)).Normal (IsGal …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
  · let f : L → IntermediateField k K := fun x => IntermediateField.lift <|
      IntermediateField.fixedField <| Subgroup.map (restrictNormalHom
      (adjoin k {x.1})) L.fixingSubgroup
    have h' (x : K) : (Subgroup.map (restrictNormalHom
      (adjoin k {x})) L.fixingSubgroup).Normal :=
      Subgroup.Normal.map h (restrictNormalHom (adjoin k {x})) (restrictNormalHom_surjective K)
    have n' (l : L) : IsGalois k (IntermediateField.fixedField <| Subgroup.map
      (restrictNormalHom (adjoin k {l.1})) L.fixingSubgroup) := by
      let _ := IsGalois.of_fixedField_normal_subgroup (Subgroup.map (restrictNormalHom
        (adjoin k {l.1})) L.fixingSubgroup)
      let cH := (Subgroup.map (restrictNormalHom (adjoin k {l.1})) L.fixingSubgroup)
      exact IsGalois.of_algEquiv <| IntermediateField.liftAlgEquiv (IntermediateField.fixedField cH)
    /-
      case refine_1
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      L : IntermediateField k K
      inst✝ : IsGalois k K
      h : (↑(InfiniteGalois.IntermediateFieldEquivClosedSubgroup L)).Normal
      f : (Subtype fun x => Membership.mem L x) → IntermediateField k K := fun x =>  …
      h' : ∀ (x : K), (Subgroup.map (AlgEquiv.restrictNormalHom (Subtype fun x_1 =>  …
      n' : ∀ (l : Subtype fun x => Membership.mem L x), IsGalois k (Subtype fun x => …
      ⊢ IsGalois k (Subtype fun x => Membership.mem L x)
    -/
    have n : Normal k ↥(⨆ (l : L), f l) := IntermediateField.normal_iSup k K f
    have : (⨆ (l : L), f l) = L := by
      apply le_antisymm
      · apply iSup_le
        intro l
        simpa only [f, ← restrict_fixedField L.fixingSubgroup (adjoin k {l.1}),
          fixedField_fixingSubgroup L] using inf_le_left
      · intro l hl
        apply le_iSup f ⟨l,hl⟩
        simpa only [f, ← restrict_fixedField L.fixingSubgroup (adjoin k {l}),
          fixedField_fixingSubgroup L, IntermediateField.mem_inf, hl, true_and]
          using adjoin_simple_le_iff.mp le_rfl
    /-
      case refine_1
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      L : IntermediateField k K
      inst✝ : IsGalois k K
      h : (↑(InfiniteGalois.IntermediateFieldEquivClosedSubgroup L)).Normal
      f : (Subtype fun x => Membership.mem L x) → IntermediateField k K := fun x =>  …
      h' : ∀ (x : K), (Subgroup.map (AlgEquiv.restrictNormalHom (Subtype fun x_1 =>  …
      n' : ∀ (l : Subtype fun x => Membership.mem L x), IsGalois k (Subtype fun x => …
      n : Normal k (Subtype fun x => Membership.mem (iSup fun l => f l) x)
      this : Eq (iSup fun l => f l) L
      ⊢ IsGalois k (Subtype fun x => Membership.mem L x)
    -/
    rw [this] at n
    /-
      case refine_1
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      L : IntermediateField k K
      inst✝ : IsGalois k K
      h : (↑(InfiniteGalois.IntermediateFieldEquivClosedSubgroup L)).Normal
      f : (Subtype fun x => Membership.mem L x) → IntermediateField k K := fun x =>  …
      h' : ∀ (x : K), (Subgroup.map (AlgEquiv.restrictNormalHom (Subtype fun x_1 =>  …
      n' : ∀ (l : Subtype fun x => Membership.mem L x), IsGalois k (Subtype fun x => …
      n : Normal k (Subtype fun x => Membership.mem L x)
      this : Eq (iSup fun l => f l) L
      ⊢ IsGalois k (Subtype fun x => Membership.mem L x)
    -/
    let _ : Algebra.IsSeparable k L := Algebra.isSeparable_tower_bot_of_isSeparable k L K
    /-
      case refine_1
      k : Type u_1
      K : Type u_2
      inst✝³ : Field k
      inst✝² : Field K
      inst✝¹ : Algebra k K
      L : IntermediateField k K
      inst✝ : IsGalois k K
      h : (↑(InfiniteGalois.IntermediateFieldEquivClosedSubgroup L)).Normal
      f : (Subtype fun x => Membership.mem L x) → IntermediateField k K := fun x =>  …
      h' : ∀ (x : K), (Subgroup.map (AlgEquiv.restrictNormalHom (Subtype fun x_1 =>  …
      n' : ∀ (l : Subtype fun x => Membership.mem L x), IsGalois k (Subtype fun x => …
      n : Normal k (Subtype fun x => Membership.mem L x)
      this : Eq (iSup fun l => f l) L
      x✝ : Algebra.IsSeparable k (Subtype fun x => Membership.mem L x) := Algebra.is …
      ⊢ IsGalois k (Subtype fun x => Membership.mem L x)
    -/
    apply IsGalois.mk
    /-
      🎉 no goals
    -/
  · simpa only [IntermediateFieldEquivClosedSubgroup, RelIso.coe_fn_mk, Equiv.coe_fn_mk,
      ← L.restrictNormalHom_ker] using MonoidHom.normal_ker (restrictNormalHom L)


