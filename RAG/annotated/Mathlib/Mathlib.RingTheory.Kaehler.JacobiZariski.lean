/--
Given `R[X] → S` and `S[Y] → T`, this is the lift of an element in `ker(S[Y] → T)`
to `ker(R[X][Y] → S[Y] → T)` constructed from `P.σ`.
-/
noncomputable
def kerCompPreimage (x : Q.ker) :
    (Q.comp P).ker := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    P✝ : Algebra.Generators R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Subtype fun x => Membership.mem Q.ker x
    ⊢ Subtype fun x => Membership.mem (Q.comp P).ker x
  -/
  refine ⟨x.1.sum fun n r ↦ ?_, ?_⟩
  · -- The use of `refine` is intentional to control the elaboration order
    -- so that the term has type `(Q.comp P).Ring` and not `MvPolynomial (Q.vars ⊕ P.vars) R`
    /-
      case refine_1
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      P✝ : Algebra.Generators R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Subtype fun x => Membership.mem Q.ker x
      n : Finsupp Q.vars Nat
      r : S
      ⊢ (Q.comp P).Ring
    -/
    refine rename ?_ (P.σ r) * monomial ?_ 1
    /-
      case refine_1.refine_1
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      P✝ : Algebra.Generators R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Subtype fun x => Membership.mem Q.ker x
      n : Finsupp Q.vars Nat
      r : S
      ⊢ P.vars → (Q.comp P).vars
    -/
    exacts [Sum.inr, n.mapDomain Sum.inl]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      P✝ : Algebra.Generators R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Subtype fun x => Membership.mem Q.ker x
      ⊢ Membership.mem (Q.comp P).ker (Finsupp.sum ↑x fun n r => HMul.hMul ((MvPolyn …
    -/
  · simp only [ker_eq_ker_aeval_val, RingHom.mem_ker]
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      P✝ : Algebra.Generators R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Subtype fun x => Membership.mem Q.ker x
      ⊢ Eq ((MvPolynomial.aeval (Q.comp P).val) (Finsupp.sum ↑x fun n r => HMul.hMul …
    -/
    conv_rhs => rw [← aeval_val_eq_zero x.2, ← x.1.support_sum_monomial_coeff]
    simp only [Finsupp.sum, map_sum, map_mul, aeval_rename, Function.comp_def, comp_val,
      Sum.elim_inr, aeval_monomial, map_one, Finsupp.prod_mapDomain_index_inj Sum.inl_injective,
      Sum.elim_inl, one_mul]
    /-
      case refine_2
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      P✝ : Algebra.Generators R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Subtype fun x => Membership.mem Q.ker x
      ⊢ Eq ((↑x).support.sum fun x_1 => HMul.hMul ((MvPolynomial.aeval fun x => (alg …
    -/
    congr! with v i
    /-
      case refine_2.a.h.e'_5
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      P✝ : Algebra.Generators R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Subtype fun x => Membership.mem Q.ker x
      v : Finsupp Q.vars Nat
      i : Membership.mem (MvPolynomial.support ↑x) v
      ⊢ Eq ((MvPolynomial.aeval fun x => (algebraMap S T) (P.val x)) (P.σ (↑x v))) ( …
    -/
    simp_rw [← IsScalarTower.toAlgHom_apply R, ← comp_aeval, AlgHom.comp_apply, P.aeval_val_σ]
    /-
      case refine_2.a.h.e'_5
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      P✝ : Algebra.Generators R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Subtype fun x => Membership.mem Q.ker x
      v : Finsupp Q.vars Nat
      i : Membership.mem (MvPolynomial.support ↑x) v
      ⊢ Eq ((IsScalarTower.toAlgHom R S T) (↑x v)) ((IsScalarTower.toAlgHom R S T) ( …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma ofComp_kerCompPreimage (x : Q.ker) :
    (Q.ofComp P).toAlgHom (kerCompPreimage Q P x) = x := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Subtype fun x => Membership.mem Q.ker x
    ⊢ Eq ((Q.ofComp P).toAlgHom ↑(Q.kerCompPreimage P x)) ↑x
  -/
  conv_rhs => rw [← x.1.support_sum_monomial_coeff]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Subtype fun x => Membership.mem Q.ker x
    ⊢ Eq ((Q.ofComp P).toAlgHom ↑(Q.kerCompPreimage P x)) ((MvPolynomial.support ↑ …
  -/
  rw [kerCompPreimage, map_finsupp_sum, Finsupp.sum]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Subtype fun x => Membership.mem Q.ker x
    ⊢ Eq ((↑x).support.sum fun a => (Q.ofComp P).toAlgHom (HMul.hMul ((MvPolynomia …
  -/
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Subtype fun x => Membership.mem Q.ker x
    j : Finsupp Q.vars Nat
    x✝ : Membership.mem (MvPolynomial.support ↑x) j
    ⊢ Eq ((Q.ofComp P).toAlgHom (HMul.hMul ((MvPolynomial.rename Sum.inr) (P.σ (↑x …
  -/
  simp only [AlgHom.toLinearMap_apply, _root_.map_mul, Hom.toAlgHom_monomial]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Subtype fun x => Membership.mem Q.ker x
    j : Finsupp Q.vars Nat
    x✝ : Membership.mem (MvPolynomial.support ↑x) j
    ⊢ Eq (HMul.hMul ((Q.ofComp P).toAlgHom ((MvPolynomial.rename Sum.inr) (P.σ (↑x …
  -/
  rw [one_smul, Finsupp.prod_mapDomain_index_inj Sum.inl_injective]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Subtype fun x => Membership.mem Q.ker x
    j : Finsupp Q.vars Nat
    x✝ : Membership.mem (MvPolynomial.support ↑x) j
    ⊢ Eq (HMul.hMul ((Q.ofComp P).toAlgHom ((MvPolynomial.rename Sum.inr) (P.σ (↑x …
  -/
  rw [rename, ← AlgHom.comp_apply, comp_aeval]
  simp only [ofComp_val, Sum.elim_inr, Function.comp_apply, self_val, id_eq,
    Sum.elim_inl, monomial_eq, Hom.toAlgHom_X]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Subtype fun x => Membership.mem Q.ker x
    j : Finsupp Q.vars Nat
    x✝ : Membership.mem (MvPolynomial.support ↑x) j
    ⊢ Eq (HMul.hMul ((MvPolynomial.aeval fun i => MvPolynomial.C (P.val i)) (P.σ ( …
  -/
  congr 1
  rw [aeval_def, IsScalarTower.algebraMap_eq R S, ← MvPolynomial.algebraMap_eq,
    ← coe_eval₂Hom, ← map_aeval, P.aeval_val_σ]
  /-
    case e_a
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Subtype fun x => Membership.mem Q.ker x
    j : Finsupp Q.vars Nat
    x✝ : Membership.mem (MvPolynomial.support ↑x) j
    ⊢ Eq ((algebraMap S Q.Ring) (↑x j)) ((algebraMap S (MvPolynomial Q.vars S)) (M …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma Cotangent.map_ofComp_ker :
    Submodule.map (Q.ofComp P).toAlgHom.toLinearMap ((Q.comp P).ker.restrictScalars R) =
      Q.ker.restrictScalars R := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    ⊢ Eq (Submodule.map (Q.ofComp P).toAlgHom.toLinearMap (Submodule.restrictScala …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      ⊢ LE.le (Submodule.map (Q.ofComp P).toAlgHom.toLinearMap (Submodule.restrictSc …
    -/
  · rintro _ ⟨x, hx, rfl⟩
    simp only [ker_eq_ker_aeval_val, Submodule.coe_restrictScalars, SetLike.mem_coe,
      RingHom.mem_ker, AlgHom.toLinearMap_apply, Submodule.restrictScalars_mem] at hx ⊢
    /-
      case a.intro.intro
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : (Q.comp P).Ring
      hx : Eq ((MvPolynomial.aeval (Q.comp P).val) x) 0
      ⊢ Eq ((MvPolynomial.aeval Q.val) ((Q.ofComp P).toAlgHom x)) 0
    -/
    rw [← hx, Hom.algebraMap_toAlgHom]
    /-
      case a.intro.intro
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : (Q.comp P).Ring
      hx : Eq ((MvPolynomial.aeval (Q.comp P).val) x) 0
      ⊢ Eq ((algebraMap T T) ((MvPolynomial.aeval (Q.comp P).val) x)) ((MvPolynomial …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      ⊢ LE.le (Submodule.restrictScalars R Q.ker) (Submodule.map (Q.ofComp P).toAlgH …
    -/
  · intro x hx
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Q.Ring
      hx : Membership.mem (Submodule.restrictScalars R Q.ker) x
      ⊢ Membership.mem (Submodule.map (Q.ofComp P).toAlgHom.toLinearMap (Submodule.r …
    -/
    exact ⟨_, (kerCompPreimage Q P ⟨x, hx⟩).2, ofComp_kerCompPreimage Q P ⟨x, hx⟩⟩
    /-
      🎉 no goals
    -/


lemma Cotangent.surjective_map_ofComp :
    Function.Surjective (Extension.Cotangent.map (Q.ofComp P).toExtensionHom) := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    ⊢ Function.Surjective ⇑(Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensi …
  -/
  intro x
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Q.toExtension.Cotangent
    ⊢ Exists fun a => Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensio …
  -/
  obtain ⟨⟨x, hx⟩, rfl⟩ := Extension.Cotangent.mk_surjective x
  /-
    case intro.mk
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Q.toExtension.Ring
    hx : Membership.mem Q.toExtension.ker x
    ⊢ Exists fun a => Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensio …
  -/
  have : x ∈ Q.ker.restrictScalars R := hx
  /-
    case intro.mk
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Q.toExtension.Ring
    hx : Membership.mem Q.toExtension.ker x
    this : Membership.mem (Submodule.restrictScalars R Q.ker) x
    ⊢ Exists fun a => Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensio …
  -/
  rw [← map_ofComp_ker Q P] at this
  /-
    case intro.mk
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Q.toExtension.Ring
    hx : Membership.mem Q.toExtension.ker x
    this : Membership.mem (Submodule.map (Q.ofComp P).toAlgHom.toLinearMap (Submod …
    ⊢ Exists fun a => Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensio …
  -/
  obtain ⟨x, hx', rfl⟩ := this
  /-
    case intro.mk.intro.intro
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : (Q.comp P).Ring
    hx' : Membership.mem (↑(Submodule.restrictScalars R (Q.comp P).ker)) x
    hx : Membership.mem Q.toExtension.ker ((Q.ofComp P).toAlgHom.toLinearMap x)
    ⊢ Exists fun a => Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensio …
  -/
  exact ⟨.mk ⟨x, hx'⟩, Extension.Cotangent.map_mk _ _⟩
  /-
    🎉 no goals
  -/


open Extension.Cotangent in
lemma Cotangent.exact :
    Function.Exact
      ((Extension.Cotangent.map (Q.toComp P).toExtensionHom).liftBaseChange T)
      (Extension.Cotangent.map (Q.ofComp P).toExtensionHom) := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    ⊢ Function.Exact ⇑(LinearMap.liftBaseChange T (Algebra.Extension.Cotangent.map …
  -/
  apply LinearMap.exact_of_comp_of_mem_range
  · rw [LinearMap.liftBaseChange_comp, ← Extension.Cotangent.map_comp,
      EmbeddingLike.map_eq_zero_iff]
    /-
      case h1
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      ⊢ Eq (Algebra.Extension.Cotangent.map ((Q.ofComp P).toExtensionHom.comp (Q.toC …
    -/
    ext x
    /-
      case h1.h.e
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : P.toExtension.Cotangent
      ⊢ Eq ((Algebra.Extension.Cotangent.map ((Q.ofComp P).toExtensionHom.comp (Q.to …
    -/
    obtain ⟨⟨x, hx⟩, rfl⟩ := Extension.Cotangent.mk_surjective x
    /-
      case h1.h.e.intro.mk
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : P.toExtension.Ring
      hx : Membership.mem P.toExtension.ker x
      ⊢ Eq ((Algebra.Extension.Cotangent.map ((Q.ofComp P).toExtensionHom.comp (Q.to …
    -/
    simp only [map_mk, Hom.toAlgHom_comp_apply, val_mk, LinearMap.zero_apply, val_zero]
    /-
      case h1.h.e.intro.mk
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : P.toExtension.Ring
      hx : Membership.mem P.toExtension.ker x
      ⊢ Eq (Q.toExtension.ker.toCotangent ⟨((Q.ofComp P).toExtensionHom.comp (Q.toCo …
    -/
    convert Q.ker.toCotangent.map_zero
    /-
      case h.e'_2.h.e'_6.h.h.e'_3
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : P.toExtension.Ring
      hx : Membership.mem P.toExtension.ker x
      e_2✝ : Eq (Subtype fun x => Membership.mem Q.toExtension.ker x) (Subtype fun x …
      ⊢ Eq (((Q.ofComp P).toExtensionHom.comp (Q.toComp P).toExtensionHom).toAlgHom  …
    -/
    trans ((IsScalarTower.toAlgHom R _ _).comp (IsScalarTower.toAlgHom R P.Ring S)) x
      /-
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type uT
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        x : P.toExtension.Ring
        hx : Membership.mem P.toExtension.ker x
        e_2✝ : Eq (Subtype fun x => Membership.mem Q.toExtension.ker x) (Subtype fun x …
        ⊢ Eq (((Q.ofComp P).toExtensionHom.comp (Q.toComp P).toExtensionHom).toAlgHom  …
      -/
    · congr
      /-
        case e_a
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type uT
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        x : P.toExtension.Ring
        hx : Membership.mem P.toExtension.ker x
        e_2✝ : Eq (Subtype fun x => Membership.mem Q.toExtension.ker x) (Subtype fun x …
        ⊢ Eq ((Q.ofComp P).toExtensionHom.comp (Q.toComp P).toExtensionHom).toAlgHom ( …
      -/
      refine MvPolynomial.algHom_ext fun i ↦ ?_
      /-
        case e_a
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type uT
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        x : P.toExtension.Ring
        hx : Membership.mem P.toExtension.ker x
        e_2✝ : Eq (Subtype fun x => Membership.mem Q.toExtension.ker x) (Subtype fun x …
        i : P.vars
        ⊢ Eq (((Q.ofComp P).toExtensionHom.comp (Q.toComp P).toExtensionHom).toAlgHom  …
      -/
      show (Q.ofComp P).toAlgHom ((Q.toComp P).toAlgHom (X i)) = _
      /-
        case e_a
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type uT
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        x : P.toExtension.Ring
        hx : Membership.mem P.toExtension.ker x
        e_2✝ : Eq (Subtype fun x => Membership.mem Q.toExtension.ker x) (Subtype fun x …
        i : P.vars
        ⊢ Eq ((Q.ofComp P).toAlgHom ((Q.toComp P).toAlgHom (MvPolynomial.X i))) (((IsS …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type uT
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        x : P.toExtension.Ring
        hx : Membership.mem P.toExtension.ker x
        e_2✝ : Eq (Subtype fun x => Membership.mem Q.toExtension.ker x) (Subtype fun x …
        ⊢ Eq (((IsScalarTower.toAlgHom R S Q.toExtension.Ring).comp (IsScalarTower.toA …
      -/
    · simp [-self_vars, aeval_val_eq_zero hx]
      /-
        🎉 no goals
      -/
    /-
      case h2
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      ⊢ ∀ (x : (Q.comp P).toExtension.Cotangent), Eq ((Algebra.Extension.Cotangent.m …
    -/
  · intro x hx
    /-
      case h2
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : (Q.comp P).toExtension.Cotangent
      hx : Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom) x) 0
      ⊢ Membership.mem (LinearMap.range (LinearMap.liftBaseChange T (Algebra.Extensi …
    -/
    obtain ⟨⟨x : (Q.comp P).Ring, hx'⟩, rfl⟩ := Extension.Cotangent.mk_surjective x
    replace hx : (Q.ofComp P).toAlgHom x ∈ Q.ker ^ 2 := by
      simpa only [map_mk, val_mk, val_zero, Ideal.toCotangent_eq_zero] using congr(($hx).val)
    rw [← Submodule.restrictScalars_mem R, pow_two, Submodule.restrictScalars_mul,
      ← map_ofComp_ker (P := P), ← Submodule.map_mul, ← Submodule.restrictScalars_mul] at hx
    /-
      case h2.intro.mk
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : (Q.comp P).Ring
      hx' : Membership.mem (Q.comp P).toExtension.ker x
      hx : Membership.mem (Submodule.map (Q.ofComp P).toAlgHom.toLinearMap (Submodul …
      ⊢ Membership.mem (LinearMap.range (LinearMap.liftBaseChange T (Algebra.Extensi …
    -/
    obtain ⟨y, hy, e⟩ := hx
    rw [AlgHom.toLinearMap_apply, eq_comm, ← sub_eq_zero, ← map_sub, ← RingHom.mem_ker,
      ← map_toComp_ker] at e
    /-
      case h2.intro.mk.intro.intro
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : (Q.comp P).Ring
      hx' : Membership.mem (Q.comp P).toExtension.ker x
      y : (Q.comp P).Ring
      hy : Membership.mem (↑(Submodule.restrictScalars R (HMul.hMul (Q.comp P).ker ( …
      e : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (HSub.hSu …
      ⊢ Membership.mem (LinearMap.range (LinearMap.liftBaseChange T (Algebra.Extensi …
    -/
    rw [LinearMap.range_liftBaseChange]
    /-
      case h2.intro.mk.intro.intro
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : (Q.comp P).Ring
      hx' : Membership.mem (Q.comp P).toExtension.ker x
      y : (Q.comp P).Ring
      hy : Membership.mem (↑(Submodule.restrictScalars R (HMul.hMul (Q.comp P).ker ( …
      e : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (HSub.hSu …
      ⊢ Membership.mem (Submodule.span T ↑(LinearMap.range (Algebra.Extension.Cotang …
    -/
    let z : (Q.comp P).ker := ⟨x - y, Ideal.sub_mem _ hx' (Ideal.mul_le_left hy)⟩
    /-
      case h2.intro.mk.intro.intro
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : (Q.comp P).Ring
      hx' : Membership.mem (Q.comp P).toExtension.ker x
      y : (Q.comp P).Ring
      hy : Membership.mem (↑(Submodule.restrictScalars R (HMul.hMul (Q.comp P).ker ( …
      e : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (HSub.hSu …
      z : Subtype fun x => Membership.mem (Q.comp P).ker x := ⟨HSub.hSub x y, ⋯⟩
      ⊢ Membership.mem (Submodule.span T ↑(LinearMap.range (Algebra.Extension.Cotang …
    -/
    have hz : z.1 ∈ P.ker.map (Q.toComp P).toAlgHom.toRingHom := e
    have : Extension.Cotangent.mk (P := (Q.comp P).toExtension) ⟨x, hx'⟩ =
      Extension.Cotangent.mk z := by
      ext; simpa only [comp_vars, val_mk, Ideal.toCotangent_eq, sub_sub_cancel, pow_two]
    rw [this, ← Submodule.restrictScalars_mem (Q.comp P).Ring, ← Submodule.mem_comap,
      ← Submodule.span_singleton_le_iff_mem, ← Submodule.map_le_map_iff_of_injective
      (f := Submodule.subtype _) Subtype.val_injective, Submodule.map_subtype_span_singleton,
      Submodule.span_singleton_le_iff_mem]
    /-
      case h2.intro.mk.intro.intro
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : (Q.comp P).Ring
      hx' : Membership.mem (Q.comp P).toExtension.ker x
      y : (Q.comp P).Ring
      hy : Membership.mem (↑(Submodule.restrictScalars R (HMul.hMul (Q.comp P).ker ( …
      e : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (HSub.hSu …
      z : Subtype fun x => Membership.mem (Q.comp P).ker x := ⟨HSub.hSub x y, ⋯⟩
      hz : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) ↑z
      this : Eq (Algebra.Extension.Cotangent.mk ⟨x, hx'⟩) (Algebra.Extension.Cotange …
      ⊢ Membership.mem (Submodule.map (Submodule.subtype (Q.comp P).toExtension.ker) …
    -/
    refine (show Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker ≤ _ from ?_) hz
    /-
      case h2.intro.mk.intro.intro
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : (Q.comp P).Ring
      hx' : Membership.mem (Q.comp P).toExtension.ker x
      y : (Q.comp P).Ring
      hy : Membership.mem (↑(Submodule.restrictScalars R (HMul.hMul (Q.comp P).ker ( …
      e : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (HSub.hSu …
      z : Subtype fun x => Membership.mem (Q.comp P).ker x := ⟨HSub.hSub x y, ⋯⟩
      hz : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) ↑z
      this : Eq (Algebra.Extension.Cotangent.mk ⟨x, hx'⟩) (Algebra.Extension.Cotange …
      ⊢ LE.le (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (Submodule.map (Subm …
    -/
    rw [Ideal.map_le_iff_le_comap]
    /-
      case h2.intro.mk.intro.intro
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : (Q.comp P).Ring
      hx' : Membership.mem (Q.comp P).toExtension.ker x
      y : (Q.comp P).Ring
      hy : Membership.mem (↑(Submodule.restrictScalars R (HMul.hMul (Q.comp P).ker ( …
      e : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (HSub.hSu …
      z : Subtype fun x => Membership.mem (Q.comp P).ker x := ⟨HSub.hSub x y, ⋯⟩
      hz : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) ↑z
      this : Eq (Algebra.Extension.Cotangent.mk ⟨x, hx'⟩) (Algebra.Extension.Cotange …
      ⊢ LE.le P.ker (Ideal.comap (Q.toComp P).toAlgHom.toRingHom (Submodule.map (Sub …
    -/
    rintro w hw
    simp only [AlgHom.toRingHom_eq_coe, Ideal.mem_comap, RingHom.coe_coe,
      Submodule.mem_map, Submodule.mem_comap, Submodule.restrictScalars_mem, Submodule.coe_subtype,
      Subtype.exists, exists_and_right, exists_eq_right,
      toExtension_Ring, toExtension_commRing, toExtension_algebra₂]
    /-
      case h2.intro.mk.intro.intro
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : (Q.comp P).Ring
      hx' : Membership.mem (Q.comp P).toExtension.ker x
      y : (Q.comp P).Ring
      hy : Membership.mem (↑(Submodule.restrictScalars R (HMul.hMul (Q.comp P).ker ( …
      e : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (HSub.hSu …
      z : Subtype fun x => Membership.mem (Q.comp P).ker x := ⟨HSub.hSub x y, ⋯⟩
      hz : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) ↑z
      this : Eq (Algebra.Extension.Cotangent.mk ⟨x, hx'⟩) (Algebra.Extension.Cotange …
      w : P.Ring
      hw : Membership.mem P.ker w
      ⊢ Exists fun x => Membership.mem (Submodule.span T ↑(LinearMap.range (Algebra. …
    -/
    refine ⟨?_, Submodule.subset_span ⟨Extension.Cotangent.mk ⟨w, hw⟩, ?_⟩⟩
      /-
        case h2.intro.mk.intro.intro.refine_1
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type uT
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        x : (Q.comp P).Ring
        hx' : Membership.mem (Q.comp P).toExtension.ker x
        y : (Q.comp P).Ring
        hy : Membership.mem (↑(Submodule.restrictScalars R (HMul.hMul (Q.comp P).ker ( …
        e : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (HSub.hSu …
        z : Subtype fun x => Membership.mem (Q.comp P).ker x := ⟨HSub.hSub x y, ⋯⟩
        hz : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) ↑z
        this : Eq (Algebra.Extension.Cotangent.mk ⟨x, hx'⟩) (Algebra.Extension.Cotange …
        w : P.Ring
        hw : Membership.mem P.ker w
        ⊢ Membership.mem (Q.comp P).toExtension.ker ((Q.toComp P).toAlgHom w)
      -/
    · simp only [ker_eq_ker_aeval_val, RingHom.mem_ker, Hom.algebraMap_toAlgHom]
      /-
        case h2.intro.mk.intro.intro.refine_1
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type uT
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        x : (Q.comp P).Ring
        hx' : Membership.mem (Q.comp P).toExtension.ker x
        y : (Q.comp P).Ring
        hy : Membership.mem (↑(Submodule.restrictScalars R (HMul.hMul (Q.comp P).ker ( …
        e : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (HSub.hSu …
        z : Subtype fun x => Membership.mem (Q.comp P).ker x := ⟨HSub.hSub x y, ⋯⟩
        hz : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) ↑z
        this : Eq (Algebra.Extension.Cotangent.mk ⟨x, hx'⟩) (Algebra.Extension.Cotange …
        w : P.Ring
        hw : Membership.mem P.ker w
        ⊢ Eq ((algebraMap S T) ((MvPolynomial.aeval P.val) w)) 0
      -/
      rw [aeval_val_eq_zero hw, map_zero]
      /-
        🎉 no goals
      -/
      /-
        case h2.intro.mk.intro.intro.refine_2
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type uT
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        x : (Q.comp P).Ring
        hx' : Membership.mem (Q.comp P).toExtension.ker x
        y : (Q.comp P).Ring
        hy : Membership.mem (↑(Submodule.restrictScalars R (HMul.hMul (Q.comp P).ker ( …
        e : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (HSub.hSu …
        z : Subtype fun x => Membership.mem (Q.comp P).ker x := ⟨HSub.hSub x y, ⋯⟩
        hz : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) ↑z
        this : Eq (Algebra.Extension.Cotangent.mk ⟨x, hx'⟩) (Algebra.Extension.Cotange …
        w : P.Ring
        hw : Membership.mem P.ker w
        ⊢ Eq ((Algebra.Extension.Cotangent.map (Q.toComp P).toExtensionHom) (Algebra.E …
      -/
    · rw [map_mk]
      /-
        case h2.intro.mk.intro.intro.refine_2
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type uT
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        x : (Q.comp P).Ring
        hx' : Membership.mem (Q.comp P).toExtension.ker x
        y : (Q.comp P).Ring
        hy : Membership.mem (↑(Submodule.restrictScalars R (HMul.hMul (Q.comp P).ker ( …
        e : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) (HSub.hSu …
        z : Subtype fun x => Membership.mem (Q.comp P).ker x := ⟨HSub.hSub x y, ⋯⟩
        hz : Membership.mem (Ideal.map (Q.toComp P).toAlgHom.toRingHom P.ker) ↑z
        this : Eq (Algebra.Extension.Cotangent.mk ⟨x, hx'⟩) (Algebra.Extension.Cotange …
        w : P.Ring
        hw : Membership.mem P.ker w
        ⊢ Eq (Algebra.Extension.Cotangent.mk ⟨(Q.toComp P).toExtensionHom.toAlgHom ↑⟨w …
      -/
      rfl
      /-
        🎉 no goals
      -/


/-- Given `R[X] → S` and `S[Y] → T`, the cotangent space of `R[X][Y] → T` is isomorphic
to the direct product of the cotangent space of `S[Y] → T` and `R[X] → S` (base changed to `T`). -/
noncomputable
def CotangentSpace.compEquiv (Q : Generators.{w} S T) (P : Generators.{w'} R S) :
    (Q.comp P).toExtension.CotangentSpace ≃ₗ[T]
      Q.toExtension.CotangentSpace × (T ⊗[S] P.toExtension.CotangentSpace) :=
  (Q.comp P).cotangentSpaceBasis.repr.trans
    (Q.cotangentSpaceBasis.prod (P.cotangentSpaceBasis.baseChange T)).repr.symm


lemma CotangentSpace.compEquiv_symm_inr :
    (compEquiv Q P).symm.toLinearMap ∘ₗ
      LinearMap.inr T Q.toExtension.CotangentSpace (T ⊗[S] P.toExtension.CotangentSpace) =
        (Extension.CotangentSpace.map (Q.toComp P).toExtensionHom).liftBaseChange T := by
  classical
  apply (P.cotangentSpaceBasis.baseChange T).ext
  intro i
  apply (Q.comp P).cotangentSpaceBasis.repr.injective
  ext j
  simp only [compEquiv, LinearEquiv.trans_symm, LinearEquiv.symm_symm,
    Basis.baseChange_apply, LinearMap.coe_comp, LinearEquiv.coe_coe, LinearMap.coe_inr,
    Function.comp_apply, LinearEquiv.trans_apply, Basis.repr_symm_apply, pderiv_X, toComp_val,
    Basis.repr_linearCombination, LinearMap.liftBaseChange_tmul, one_smul, repr_CotangentSpaceMap]
  obtain (j | j) := j <;>
    simp only [comp_vars, Basis.prod_repr_inr, Basis.baseChange_repr_tmul,
      Basis.repr_self, Basis.prod_repr_inl, map_zero, Finsupp.coe_zero,
      Pi.zero_apply, ne_eq, not_false_eq_true, Pi.single_eq_of_ne, Pi.single_apply,
      Finsupp.single_apply, ite_smul, one_smul, zero_smul, Sum.inr.injEq,
        RingHom.map_ite_one_zero, reduceCtorEq, ↓reduceIte]


lemma CotangentSpace.compEquiv_symm_zero (x) :
    (compEquiv Q P).symm (0, x) =
        (Extension.CotangentSpace.map (Q.toComp P).toExtensionHom).liftBaseChange T x :=
  DFunLike.congr_fun (compEquiv_symm_inr Q P) x


lemma CotangentSpace.fst_compEquiv :
    LinearMap.fst T Q.toExtension.CotangentSpace (T ⊗[S] P.toExtension.CotangentSpace) ∘ₗ
      (compEquiv Q P).toLinearMap = Extension.CotangentSpace.map (Q.ofComp P).toExtensionHom := by
  classical
  apply (Q.comp P).cotangentSpaceBasis.ext
  intro i
  apply Q.cotangentSpaceBasis.repr.injective
  ext j
  simp only [compEquiv, LinearMap.coe_comp, LinearEquiv.coe_coe, Function.comp_apply, ofComp_val,
    LinearEquiv.trans_apply, Basis.repr_self, LinearMap.fst_apply, repr_CotangentSpaceMap]
  obtain (i | i) := i <;>
    simp only [comp_vars, Basis.repr_symm_apply, Finsupp.linearCombination_single, Basis.prod_apply,
      LinearMap.coe_inl, LinearMap.coe_inr, Sum.elim_inl, Function.comp_apply, one_smul,
      Basis.repr_self, Finsupp.single_apply, pderiv_X, Pi.single_apply, RingHom.map_ite_one_zero,
      Sum.elim_inr, Function.comp_apply, Basis.baseChange_apply, one_smul,
      map_zero, Finsupp.coe_zero, Pi.zero_apply, derivation_C]


lemma CotangentSpace.fst_compEquiv_apply (x) :
    (compEquiv Q P x).1 = Extension.CotangentSpace.map (Q.ofComp P).toExtensionHom x :=
  DFunLike.congr_fun (fst_compEquiv Q P) x


lemma CotangentSpace.map_toComp_injective :
    Function.Injective
      ((Extension.CotangentSpace.map (Q.toComp P).toExtensionHom).liftBaseChange T) := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    ⊢ Function.Injective ⇑(LinearMap.liftBaseChange T (Algebra.Extension.Cotangent …
  -/
  rw [← compEquiv_symm_inr]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    ⊢ Function.Injective ⇑((↑(Algebra.Generators.CotangentSpace.compEquiv Q P).sym …
  -/
  apply (compEquiv Q P).symm.injective.comp
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    ⊢ Function.Injective ⇑(LinearMap.inr T Q.toExtension.CotangentSpace (TensorPro …
  -/
  exact Prod.mk.inj_left _
  /-
    🎉 no goals
  -/


lemma CotangentSpace.map_ofComp_surjective :
    Function.Surjective (Extension.CotangentSpace.map (Q.ofComp P).toExtensionHom) := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    ⊢ Function.Surjective ⇑(Algebra.Extension.CotangentSpace.map (Q.ofComp P).toEx …
  -/
  rw [← fst_compEquiv]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    ⊢ Function.Surjective ⇑((LinearMap.fst T Q.toExtension.CotangentSpace (TensorP …
  -/
  exact (Prod.fst_surjective).comp (compEquiv Q P).surjective
  /-
    🎉 no goals
  -/


lemma CotangentSpace.exact :
    Function.Exact ((Extension.CotangentSpace.map (Q.toComp P).toExtensionHom).liftBaseChange T)
      (Extension.CotangentSpace.map (Q.ofComp P).toExtensionHom) := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    ⊢ Function.Exact ⇑(LinearMap.liftBaseChange T (Algebra.Extension.CotangentSpac …
  -/
  rw [← fst_compEquiv, ← compEquiv_symm_inr]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    ⊢ Function.Exact ⇑((↑(Algebra.Generators.CotangentSpace.compEquiv Q P).symm).c …
  -/
  conv_rhs => rw [← LinearEquiv.symm_symm (compEquiv Q P)]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    ⊢ Function.Exact ⇑((↑(Algebra.Generators.CotangentSpace.compEquiv Q P).symm).c …
  -/
  rw [LinearEquiv.conj_exact_iff_exact]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    ⊢ Function.Exact ⇑(LinearMap.inr T Q.toExtension.CotangentSpace (TensorProduct …
  -/
  exact Function.Exact.inr_fst
  /-
    🎉 no goals
  -/


variable (R) in
/--
Given `0 → I → S[Y] → T → 0`, this is an auxiliary map from `S[Y]` to `T ⊗[S] Ω[S⁄R]` whose
restriction to `ker(I/I² → ⊕ S dyᵢ)` is the connecting homomorphism in the Jacobi-Zariski sequence.
-/
noncomputable
def δAux :
    Q.Ring →ₗ[R] T ⊗[S] Ω[S⁄R] :=
  Finsupp.lsum R (R := R) fun f ↦
    (TensorProduct.mk S T _ (f.prod (Q.val · ^ ·))).restrictScalars R ∘ₗ (D R S).toLinearMap


lemma δAux_monomial (n r) :
    δAux R Q (monomial n r) = (n.prod (Q.val · ^ ·)) ⊗ₜ D R S r :=
  Finsupp.lsum_single _ _ _ _


@[simp]
lemma δAux_X (i) :
    δAux R Q (X i) = 0 := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    i : Q.vars
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q) (MvPolynomial.X i)) 0
  -/
  rw [X, δAux_monomial]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    i : Q.vars
    ⊢ Eq (TensorProduct.tmul S ((Finsupp.single i 1).prod fun x1 x2 => HPow.hPow ( …
  -/
  simp only [Derivation.map_one_eq_zero, tmul_zero]
  /-
    🎉 no goals
  -/


lemma δAux_mul (x y) :
    δAux R Q (x * y) = x • (δAux R Q y) + y • (δAux R Q x) := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    x y : Q.Ring
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q) (HMul.hMul x y)) (HAdd.hAdd (H …
  -/
  induction' x using MvPolynomial.induction_on' with n r x₁ x₂ hx₁ hx₂
    /-
      case h1
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      y : Q.Ring
      n : Finsupp Q.vars Nat
      r : S
      ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q) (HMul.hMul ((MvPolynomial.mono …
    -/
  · induction' y using MvPolynomial.induction_on' with m s y₁ y₂ hy₁ hy₂
    · simp only [monomial_mul, δAux_monomial, Derivation.leibniz, tmul_add, tmul_smul,
        smul_tmul', smul_eq_mul, Algebra.smul_def, algebraMap_apply, aeval_monomial, mul_assoc]
      /-
        case h1.h1
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type uT
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        n : Finsupp Q.vars Nat
        r : S
        m : Finsupp Q.vars Nat
        s : S
        ⊢ Eq (HAdd.hAdd (TensorProduct.tmul S (HMul.hMul ((algebraMap S T) r) ((HAdd.h …
      -/
      rw [mul_comm (m.prod _) (n.prod _)]
      /-
        case h1.h1
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type uT
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        n : Finsupp Q.vars Nat
        r : S
        m : Finsupp Q.vars Nat
        s : S
        ⊢ Eq (HAdd.hAdd (TensorProduct.tmul S (HMul.hMul ((algebraMap S T) r) ((HAdd.h …
      -/
      simp only [pow_zero, implies_true, pow_add, Finsupp.prod_add_index']
      /-
        🎉 no goals
      -/
      /-
        case h1.h2
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type uT
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        n : Finsupp Q.vars Nat
        r : S
        y₁ y₂ : MvPolynomial Q.vars S
        hy₁ : Eq ((Algebra.Generators.H1Cotangent.δAux R Q) (HMul.hMul ((MvPolynomial. …
        hy₂ : Eq ((Algebra.Generators.H1Cotangent.δAux R Q) (HMul.hMul ((MvPolynomial. …
        ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q) (HMul.hMul ((MvPolynomial.mono …
      -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    · simp only [map_add, smul_add, hy₁, hy₂, mul_add, add_smul]; abel
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    /-
      case h2
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      y : Q.Ring
      x₁ x₂ : MvPolynomial Q.vars S
      hx₁ : Eq ((Algebra.Generators.H1Cotangent.δAux R Q) (HMul.hMul x₁ y)) (HAdd.hA …
      hx₂ : Eq ((Algebra.Generators.H1Cotangent.δAux R Q) (HMul.hMul x₂ y)) (HAdd.hA …
      ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q) (HMul.hMul (HAdd.hAdd x₁ x₂) y …
    -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
  · simp only [add_mul, map_add, hx₁, hx₂, add_smul, smul_add]; abel
                                                                /-
                                                                  🎉 no goals
                                                                -/


lemma δAux_C (r) :
    δAux R Q (C r) = 1 ⊗ₜ D R S r := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    r : S
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q) (MvPolynomial.C r)) (TensorPro …
  -/
  rw [← monomial_zero', δAux_monomial, Finsupp.prod_zero_index]
  /-
    🎉 no goals
  -/


lemma δAux_toAlgHom {Q : Generators.{u₁} S T}
    {Q' : Generators.{u₃} S T} (f : Hom Q Q') (x) :
    δAux R Q' (f.toAlgHom x) = δAux R Q x + Finsupp.linearCombination _ (δAux R Q' ∘ f.val)
      (Q.cotangentSpaceBasis.repr ((1 : T) ⊗ₜ[Q.Ring] D S Q.Ring x : _)) := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    Q' : Algebra.Generators S T
    f : Q.Hom Q'
    x : Q.Ring
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q') (f.toAlgHom x)) (HAdd.hAdd (( …
  -/
  letI : AddCommGroup (T ⊗[S] Ω[S⁄R]) := inferInstance
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    Q' : Algebra.Generators S T
    f : Q.Hom Q'
    x : Q.Ring
    this : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferInst …
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q') (f.toAlgHom x)) (HAdd.hAdd (( …
  -/
  have : IsScalarTower Q.Ring Q.Ring T := IsScalarTower.left _
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    Q' : Algebra.Generators S T
    f : Q.Hom Q'
    x : Q.Ring
    this✝ : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferIns …
    this : IsScalarTower Q.Ring Q.Ring T
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q') (f.toAlgHom x)) (HAdd.hAdd (( …
  -/
  induction' x using MvPolynomial.induction_on with s x₁ x₂ hx₁ hx₂ p n IH
    /-
      case h_C
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      Q' : Algebra.Generators S T
      f : Q.Hom Q'
      this✝ : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferIns …
      this : IsScalarTower Q.Ring Q.Ring T
      s : S
      ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q') (f.toAlgHom (MvPolynomial.C s …
    -/
  · simp [MvPolynomial.algebraMap_eq, δAux_C]
    /-
      🎉 no goals
    -/
    /-
      case h_add
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      Q' : Algebra.Generators S T
      f : Q.Hom Q'
      this✝ : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferIns …
      this : IsScalarTower Q.Ring Q.Ring T
      x₁ x₂ : MvPolynomial Q.vars S
      hx₁ : Eq ((Algebra.Generators.H1Cotangent.δAux R Q') (f.toAlgHom x₁)) (HAdd.hA …
      hx₂ : Eq ((Algebra.Generators.H1Cotangent.δAux R Q') (f.toAlgHom x₂)) (HAdd.hA …
      ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q') (f.toAlgHom (HAdd.hAdd x₁ x₂) …
    -/
  · simp only [map_add, hx₁, hx₂, tmul_add]
    /-
      case h_add
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      Q' : Algebra.Generators S T
      f : Q.Hom Q'
      this✝ : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferIns …
      this : IsScalarTower Q.Ring Q.Ring T
      x₁ x₂ : MvPolynomial Q.vars S
      hx₁ : Eq ((Algebra.Generators.H1Cotangent.δAux R Q') (f.toAlgHom x₁)) (HAdd.hA …
      hx₂ : Eq ((Algebra.Generators.H1Cotangent.δAux R Q') (f.toAlgHom x₂)) (HAdd.hA …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd ((Algebra.Generators.H1Cotangent.δAux R Q) x₁) ((Fi …
    -/
    rw [add_add_add_comm]
    /-
      🎉 no goals
    -/
  · simp only [map_mul, Hom.toAlgHom_X, δAux_mul, algebraMap_apply, Hom.algebraMap_toAlgHom,
      ← @IsScalarTower.algebraMap_smul Q'.Ring T, id.map_eq_id, δAux_X, RingHomCompTriple.comp_eq,
      RingHom.id_apply, coe_eval₂Hom, IH, Hom.aeval_val, smul_add, map_aeval, tmul_add, tmul_smul,
      ← @IsScalarTower.algebraMap_smul Q.Ring T, smul_zero, aeval_X, zero_add, Derivation.leibniz,
      LinearEquiv.map_add, LinearEquiv.map_smul, Basis.repr_self, LinearMap.map_add, one_smul,
      LinearMap.map_smul, Finsupp.linearCombination_single,
      Function.comp_apply, ← cotangentSpaceBasis_apply]
    /-
      case h_X
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      Q' : Algebra.Generators S T
      f : Q.Hom Q'
      this✝ : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferIns …
      this : IsScalarTower Q.Ring Q.Ring T
      p : MvPolynomial Q.vars S
      n : Q.vars
      IH : Eq ((Algebra.Generators.H1Cotangent.δAux R Q') (f.toAlgHom p)) (HAdd.hAdd …
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (MvPolynomial.eval₂ (algebraMap S T) (fun i => Q. …
    -/
    rw [add_left_comm]
    /-
      case h_X
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      Q' : Algebra.Generators S T
      f : Q.Hom Q'
      this✝ : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferIns …
      this : IsScalarTower Q.Ring Q.Ring T
      p : MvPolynomial Q.vars S
      n : Q.vars
      IH : Eq ((Algebra.Generators.H1Cotangent.δAux R Q') (f.toAlgHom p)) (HAdd.hAdd …
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Q.val n) ((Algebra.Generators.H1Cotangent.δAux R …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma δAux_ofComp (x : (Q.comp P).Ring) :
    δAux R Q ((Q.ofComp P).toAlgHom x) =
      P.toExtension.toKaehler.baseChange T (CotangentSpace.compEquiv Q P
        (1 ⊗ₜ[(Q.comp P).Ring] (D R (Q.comp P).Ring) x : _)).2 := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : (Q.comp P).Ring
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q) ((Q.ofComp P).toAlgHom x)) ((L …
  -/
  letI : AddCommGroup (T ⊗[S] Ω[S⁄R]) := inferInstance
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : (Q.comp P).Ring
    this : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferInst …
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q) ((Q.ofComp P).toAlgHom x)) ((L …
  -/
  have : IsScalarTower (Q.comp P).Ring (Q.comp P).Ring T := IsScalarTower.left _
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : (Q.comp P).Ring
    this✝ : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferIns …
    this : IsScalarTower (Q.comp P).Ring (Q.comp P).Ring T
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q) ((Q.ofComp P).toAlgHom x)) ((L …
  -/
  induction' x using MvPolynomial.induction_on with s x₁ x₂ hx₁ hx₂ p n IH
  · simp only [algHom_C, δAux_C, sub_self, derivation_C, Derivation.map_algebraMap,
      tmul_zero, map_zero, add_zero, MvPolynomial.algebraMap_apply, Prod.snd_zero]
    /-
      case h_add
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      this✝ : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferIns …
      this : IsScalarTower (Q.comp P).Ring (Q.comp P).Ring T
      x₁ x₂ : MvPolynomial (Q.comp P).vars R
      hx₁ : Eq ((Algebra.Generators.H1Cotangent.δAux R Q) ((Q.ofComp P).toAlgHom x₁) …
      hx₂ : Eq ((Algebra.Generators.H1Cotangent.δAux R Q) ((Q.ofComp P).toAlgHom x₂) …
      ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q) ((Q.ofComp P).toAlgHom (HAdd.h …
    -/
  · simp only [map_add, hx₁, hx₂, tmul_add, Prod.snd_add]
    /-
      🎉 no goals
    -/
  · simp only [map_mul, Hom.toAlgHom_X, ofComp_val, δAux_mul,
      ← @IsScalarTower.algebraMap_smul Q.Ring T, algebraMap_apply, Hom.algebraMap_toAlgHom,
      id.map_eq_id, map_aeval, RingHomCompTriple.comp_eq, comp_val, RingHom.id_apply, coe_eval₂Hom,
      IH, Derivation.leibniz, tmul_add, tmul_smul, ← cotangentSpaceBasis_apply,
      ← @IsScalarTower.algebraMap_smul (Q.comp P).Ring T, aeval_X, LinearEquiv.map_add,
      LinearMapClass.map_smul, Prod.snd_add, Prod.smul_snd, LinearMap.map_add]
    /-
      case h_X
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      this✝ : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferIns …
      this : IsScalarTower (Q.comp P).Ring (Q.comp P).Ring T
      p : MvPolynomial (Q.comp P).vars R
      n : (Q.comp P).vars
      IH : Eq ((Algebra.Generators.H1Cotangent.δAux R Q) ((Q.ofComp P).toAlgHom p))  …
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (MvPolynomial.eval₂ (algebraMap R T) (fun i => Su …
    -/
    obtain (n | n) := n
    · simp only [comp_vars, Sum.elim_inl, δAux_X, smul_zero, aeval_X,
        CotangentSpace.compEquiv, LinearEquiv.trans_apply, Basis.repr_symm_apply, zero_add,
        Basis.repr_self, Finsupp.linearCombination_single, Basis.prod_apply, LinearMap.coe_inl,
        LinearMap.coe_inr, Function.comp_apply, one_smul, map_zero]
    · simp only [comp_vars, Sum.elim_inr, Function.comp_apply, algHom_C, δAux_C,
        CotangentSpace.compEquiv, LinearEquiv.trans_apply, Basis.repr_symm_apply,
        algebraMap_smul, Basis.repr_self, Finsupp.linearCombination_single, Basis.prod_apply,
        LinearMap.coe_inr, Basis.baseChange_apply, one_smul, LinearMap.baseChange_tmul,
        toKaehler_cotangentSpaceBasis, add_left_inj, LinearMap.coe_inl]
      /-
        case h_X.inr
        R : Type u
        S : Type v
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        T : Type uT
        inst✝³ : CommRing T
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        Q : Algebra.Generators S T
        P : Algebra.Generators R S
        this✝ : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferIns …
        this : IsScalarTower (Q.comp P).Ring (Q.comp P).Ring T
        p : MvPolynomial (Q.comp P).vars R
        IH : Eq ((Algebra.Generators.H1Cotangent.δAux R Q) ((Q.ofComp P).toAlgHom p))  …
        n : P.vars
        ⊢ Eq (HSMul.hSMul (MvPolynomial.eval₂ (algebraMap R T) (fun i => Sum.elim Q.va …
      -/
      rfl
      /-
        🎉 no goals
      -/


lemma map_comp_cotangentComplex_baseChange :
    (Extension.CotangentSpace.map (Q.toComp P).toExtensionHom).liftBaseChange T ∘ₗ
      P.toExtension.cotangentComplex.baseChange T =
    (Q.comp P).toExtension.cotangentComplex ∘ₗ
      (Extension.Cotangent.map (Q.toComp P).toExtensionHom).liftBaseChange T := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    ⊢ Eq ((LinearMap.liftBaseChange T (Algebra.Extension.CotangentSpace.map (Q.toC …
  -/
  ext x; simp [Extension.CotangentSpace.map_cotangentComplex]
         /-
           🎉 no goals
         -/


open Generators in
/--
The connecting homomorphism in the Jacobi-Zariski sequence for given presentations.
Given representations `0 → I → R[X] → S → 0` and `0 → K → S[Y] → T → 0`,
we may consider the induced representation `0 → J → R[X, Y] → T → 0`,
and this map is obtained by applying snake lemma to the following diagram
```
    T ⊗[S] Ω[S/R]    →          Ω[T/R]        →   Ω[T/S]  → 0
        ↑                         ↑                 ↑
0 → T ⊗[S] (⨁ₓ S dx) → (⨁ₓ T dx) ⊕ (⨁ᵧ T dy) →  ⨁ᵧ T dy → 0
        ↑                         ↑                 ↑
    T ⊗[S] (I/I²)    →           J/J²         →    K/K²   → 0
                                  ↑                 ↑
                             H¹(L_{T/R})      → H¹(L_{T/S})

```
This is independent from the presentations chosen. See `H1Cotangent.δ_comp_equiv`.
-/
noncomputable
def δ :
    Q.toExtension.H1Cotangent →ₗ[T] T ⊗[S] Ω[S⁄R] :=
  SnakeLemma.δ'
    (P.toExtension.cotangentComplex.baseChange T)
    (Q.comp P).toExtension.cotangentComplex
    Q.toExtension.cotangentComplex
    ((Extension.Cotangent.map (toComp Q P).toExtensionHom).liftBaseChange T)
    (Extension.Cotangent.map (ofComp Q P).toExtensionHom)
    (Cotangent.exact Q P)
    ((Extension.CotangentSpace.map (toComp Q P).toExtensionHom).liftBaseChange T)
    (Extension.CotangentSpace.map (ofComp Q P).toExtensionHom)
    (CotangentSpace.exact Q P)
    (map_comp_cotangentComplex_baseChange Q P)
        /-
          R : Type u
          S : Type v
          inst✝⁶ : CommRing R
          inst✝⁵ : CommRing S
          inst✝⁴ : Algebra R S
          P✝ : Algebra.Generators R S
          T : Type uT
          inst✝³ : CommRing T
          inst✝² : Algebra R T
          inst✝¹ : Algebra S T
          inst✝ : IsScalarTower R S T
          Q : Algebra.Generators S T
          P : Algebra.Generators R S
          ⊢ Eq ((Algebra.Extension.CotangentSpace.map (Q.ofComp P).toExtensionHom).comp  …
        -/
    (by ext; exact Extension.CotangentSpace.map_cotangentComplex (ofComp Q P).toExtensionHom _)
             /-
               🎉 no goals
             -/
    Q.toExtension.h1Cotangentι
    (LinearMap.exact_subtype_ker_map _)
    (N₁ := T ⊗[S] P.toExtension.CotangentSpace)
    (P.toExtension.toKaehler.baseChange T)
    (lTensor_exact T P.toExtension.exact_cotangentComplex_toKaehler
      P.toExtension.toKaehler_surjective)
    (Cotangent.surjective_map_ofComp Q P)
    (CotangentSpace.map_toComp_injective Q P)


lemma exact_δ_map :
    Function.Exact (δ Q P) (mapBaseChange R S T) := by
  apply SnakeLemma.exact_δ_left (π₂ := (Q.comp P).toExtension.toKaehler)
    (hπ₂ := (Q.comp P).toExtension.exact_cotangentComplex_toKaehler)
    /-
      case hF
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      ⊢ Eq ((KaehlerDifferential.mapBaseChange R S T).comp (LinearMap.baseChange T P …
    -/
  · apply (P.cotangentSpaceBasis.baseChange T).ext
    /-
      case hF
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      ⊢ ∀ (i : P.vars), Eq (((KaehlerDifferential.mapBaseChange R S T).comp (LinearM …
    -/
    intro i
    simp only [Basis.baseChange_apply, LinearMap.coe_comp, Function.comp_apply,
      LinearMap.baseChange_tmul, toKaehler_cotangentSpaceBasis, mapBaseChange_tmul, map_D,
      one_smul, comp_vars, LinearMap.liftBaseChange_tmul]
    /-
      case hF
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      i : P.vars
      ⊢ Eq ((KaehlerDifferential.D R T) ((algebraMap S T) (P.val i))) ((Q.comp P).to …
    -/
    rw [cotangentSpaceBasis_apply]
    /-
      case hF
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      i : P.vars
      ⊢ Eq ((KaehlerDifferential.D R T) ((algebraMap S T) (P.val i))) ((Q.comp P).to …
    -/
    conv_rhs => enter [2]; tactic => exact Extension.CotangentSpace.map_tmul ..
    /-
      case hF
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      i : P.vars
      ⊢ Eq ((KaehlerDifferential.D R T) ((algebraMap S T) (P.val i))) ((Q.comp P).to …
    -/
    simp only [map_one, mapBaseChange_tmul, map_D, one_smul]
    /-
      case hF
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      i : P.vars
      ⊢ Eq ((KaehlerDifferential.D R T) ((algebraMap S T) (P.val i))) ((KaehlerDiffe …
    -/
    simp [Extension.Hom.toAlgHom]
    /-
      🎉 no goals
    -/
    /-
      case h
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      ⊢ Function.Surjective ⇑(LinearMap.baseChange T P.toExtension.toKaehler)
    -/
  · exact LinearMap.lTensor_surjective T P.toExtension.toKaehler_surjective
    /-
      🎉 no goals
    -/


lemma δ_eq (x : Q.toExtension.H1Cotangent) (y)
    (hy : Extension.Cotangent.map (ofComp Q P).toExtensionHom y = x.1) (z)
    (hz : (Extension.CotangentSpace.map (toComp Q P).toExtensionHom).liftBaseChange T z =
      (Q.comp P).toExtension.cotangentComplex y) :
    δ Q P x = P.toExtension.toKaehler.baseChange T z := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Q.toExtension.H1Cotangent
    y : (Q.comp P).toExtension.Cotangent
    hy : Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom) y) ↑x
    z : TensorProduct S T P.toExtension.CotangentSpace
    hz : Eq ((LinearMap.liftBaseChange T (Algebra.Extension.CotangentSpace.map (Q. …
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δ Q P) x) ((LinearMap.baseChange T P.toE …
  -/
  apply SnakeLemma.δ_eq
  /-
    case hy
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Q.toExtension.H1Cotangent
    y : (Q.comp P).toExtension.Cotangent
    hy : Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom) y) ↑x
    z : TensorProduct S T P.toExtension.CotangentSpace
    hz : Eq ((LinearMap.liftBaseChange T (Algebra.Extension.CotangentSpace.map (Q. …
    ⊢ Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom) ?y) (Algeb …
  -/
  exacts [hy, hz]
  /-
    🎉 no goals
  -/


lemma δ_eq_δAux (x : Q.ker) (hx) :
    δ Q P ⟨.mk x, hx⟩ = δAux R Q x.1 := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Subtype fun x => Membership.mem Q.ker x
    hx : Membership.mem (LinearMap.ker Q.toExtension.cotangentComplex) (Algebra.Ex …
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δ Q P) ⟨Algebra.Extension.Cotangent.mk x …
  -/
  let y := Extension.Cotangent.mk (P := (Q.comp P).toExtension) (Q.kerCompPreimage P x)
  have hy : (Extension.Cotangent.map (Q.ofComp P).toExtensionHom) y = Extension.Cotangent.mk x := by
    simp only [y, Extension.Cotangent.map_mk]
    congr
    exact ofComp_kerCompPreimage Q P x
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Subtype fun x => Membership.mem Q.ker x
    hx : Membership.mem (LinearMap.ker Q.toExtension.cotangentComplex) (Algebra.Ex …
    y : (Q.comp P).toExtension.Cotangent := Algebra.Extension.Cotangent.mk (Q.kerC …
    hy : Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom) y) (Alg …
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δ Q P) ⟨Algebra.Extension.Cotangent.mk x …
  -/
  let z := (CotangentSpace.compEquiv Q P ((Q.comp P).toExtension.cotangentComplex y)).2
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    x : Subtype fun x => Membership.mem Q.ker x
    hx : Membership.mem (LinearMap.ker Q.toExtension.cotangentComplex) (Algebra.Ex …
    y : (Q.comp P).toExtension.Cotangent := Algebra.Extension.Cotangent.mk (Q.kerC …
    hy : Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom) y) (Alg …
    z : TensorProduct S T P.toExtension.CotangentSpace := ((Algebra.Generators.Cot …
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δ Q P) ⟨Algebra.Extension.Cotangent.mk x …
  -/
  rw [H1Cotangent.δ_eq (y := y) (z := z)]
    /-
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Subtype fun x => Membership.mem Q.ker x
      hx : Membership.mem (LinearMap.ker Q.toExtension.cotangentComplex) (Algebra.Ex …
      y : (Q.comp P).toExtension.Cotangent := Algebra.Extension.Cotangent.mk (Q.kerC …
      hy : Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom) y) (Alg …
      z : TensorProduct S T P.toExtension.CotangentSpace := ((Algebra.Generators.Cot …
      ⊢ Eq ((LinearMap.baseChange T P.toExtension.toKaehler) z) ((Algebra.Generators …
    -/
  · rw [← ofComp_kerCompPreimage Q P x, δAux_ofComp]
    /-
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Subtype fun x => Membership.mem Q.ker x
      hx : Membership.mem (LinearMap.ker Q.toExtension.cotangentComplex) (Algebra.Ex …
      y : (Q.comp P).toExtension.Cotangent := Algebra.Extension.Cotangent.mk (Q.kerC …
      hy : Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom) y) (Alg …
      z : TensorProduct S T P.toExtension.CotangentSpace := ((Algebra.Generators.Cot …
      ⊢ Eq ((LinearMap.baseChange T P.toExtension.toKaehler) z) ((LinearMap.baseChan …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case hy
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Subtype fun x => Membership.mem Q.ker x
      hx : Membership.mem (LinearMap.ker Q.toExtension.cotangentComplex) (Algebra.Ex …
      y : (Q.comp P).toExtension.Cotangent := Algebra.Extension.Cotangent.mk (Q.kerC …
      hy : Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom) y) (Alg …
      z : TensorProduct S T P.toExtension.CotangentSpace := ((Algebra.Generators.Cot …
      ⊢ Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom) y) ↑⟨Algeb …
    -/
  · exact hy
    /-
      🎉 no goals
    -/
    /-
      case hz
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Subtype fun x => Membership.mem Q.ker x
      hx : Membership.mem (LinearMap.ker Q.toExtension.cotangentComplex) (Algebra.Ex …
      y : (Q.comp P).toExtension.Cotangent := Algebra.Extension.Cotangent.mk (Q.kerC …
      hy : Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom) y) (Alg …
      z : TensorProduct S T P.toExtension.CotangentSpace := ((Algebra.Generators.Cot …
      ⊢ Eq ((LinearMap.liftBaseChange T (Algebra.Extension.CotangentSpace.map (Q.toC …
    -/
  · rw [← CotangentSpace.compEquiv_symm_inr]
    /-
      case hz
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Subtype fun x => Membership.mem Q.ker x
      hx : Membership.mem (LinearMap.ker Q.toExtension.cotangentComplex) (Algebra.Ex …
      y : (Q.comp P).toExtension.Cotangent := Algebra.Extension.Cotangent.mk (Q.kerC …
      hy : Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom) y) (Alg …
      z : TensorProduct S T P.toExtension.CotangentSpace := ((Algebra.Generators.Cot …
      ⊢ Eq (((↑(Algebra.Generators.CotangentSpace.compEquiv Q P).symm).comp (LinearM …
    -/
    apply (CotangentSpace.compEquiv Q P).injective
    simp only [LinearMap.coe_comp, LinearEquiv.coe_coe, LinearMap.coe_inr, Function.comp_apply,
      LinearEquiv.apply_symm_apply, z]
    /-
      case hz.a
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Subtype fun x => Membership.mem Q.ker x
      hx : Membership.mem (LinearMap.ker Q.toExtension.cotangentComplex) (Algebra.Ex …
      y : (Q.comp P).toExtension.Cotangent := Algebra.Extension.Cotangent.mk (Q.kerC …
      hy : Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom) y) (Alg …
      z : TensorProduct S T P.toExtension.CotangentSpace := ((Algebra.Generators.Cot …
      ⊢ Eq { fst := 0, snd := ((Algebra.Generators.CotangentSpace.compEquiv Q P) ((Q …
    -/
    ext
    /-
      case hz.a.fst
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Subtype fun x => Membership.mem Q.ker x
      hx : Membership.mem (LinearMap.ker Q.toExtension.cotangentComplex) (Algebra.Ex …
      y : (Q.comp P).toExtension.Cotangent := Algebra.Extension.Cotangent.mk (Q.kerC …
      hy : Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom) y) (Alg …
      z : TensorProduct S T P.toExtension.CotangentSpace := ((Algebra.Generators.Cot …
      ⊢ Eq { fst := 0, snd := ((Algebra.Generators.CotangentSpace.compEquiv Q P) ((Q …
    -/
    swap; · rfl
            /-
              🎉 no goals
            -/
    show 0 = (LinearMap.fst T Q.toExtension.CotangentSpace (T ⊗[S] P.toExtension.CotangentSpace) ∘ₗ
      (CotangentSpace.compEquiv Q P).toLinearMap) ((Q.comp P).toExtension.cotangentComplex y)
    /-
      case hz.a.fst
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      x : Subtype fun x => Membership.mem Q.ker x
      hx : Membership.mem (LinearMap.ker Q.toExtension.cotangentComplex) (Algebra.Ex …
      y : (Q.comp P).toExtension.Cotangent := Algebra.Extension.Cotangent.mk (Q.kerC …
      hy : Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom) y) (Alg …
      z : TensorProduct S T P.toExtension.CotangentSpace := ((Algebra.Generators.Cot …
      ⊢ Eq 0 (((LinearMap.fst T Q.toExtension.CotangentSpace (TensorProduct S T P.to …
    -/
    rw [CotangentSpace.fst_compEquiv, Extension.CotangentSpace.map_cotangentComplex, hy, hx]
    /-
      🎉 no goals
    -/


lemma δ_eq_δ (Q : Generators.{u₁} S T) (P : Generators.{u₂} R S)
    (P' : Generators.{u₃} R S) :
    δ Q P = δ Q P' := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    P' : Algebra.Generators R S
    ⊢ Eq (Algebra.Generators.H1Cotangent.δ Q P) (Algebra.Generators.H1Cotangent.δ  …
  -/
  ext ⟨x, hx⟩
  /-
    case h.mk
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    P' : Algebra.Generators R S
    x : Q.toExtension.Cotangent
    hx : Membership.mem (LinearMap.ker Q.toExtension.cotangentComplex) x
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δ Q P) ⟨x, hx⟩) ((Algebra.Generators.H1C …
  -/
  obtain ⟨x, rfl⟩ := Extension.Cotangent.mk_surjective x
  /-
    case h.mk.intro
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    P' : Algebra.Generators R S
    x : Subtype fun x => Membership.mem Q.toExtension.ker x
    hx : Membership.mem (LinearMap.ker Q.toExtension.cotangentComplex) (Algebra.Ex …
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δ Q P) ⟨Algebra.Extension.Cotangent.mk x …
  -/
  rw [δ_eq_δAux, δ_eq_δAux]
  /-
    🎉 no goals
  -/


lemma exact_map_δ :
    Function.Exact (Extension.H1Cotangent.map (Q.ofComp P).toExtensionHom) (δ Q P) := by
  apply SnakeLemma.exact_δ_right
    (ι₂ := (Q.comp P).toExtension.h1Cotangentι)
    (hι₂ := LinearMap.exact_subtype_ker_map _)
    /-
      case hF
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      ⊢ Eq ((Algebra.Extension.Cotangent.map (Q.ofComp P).toExtensionHom).comp Algeb …
    -/
  · ext x; rfl
           /-
             🎉 no goals
           -/
    /-
      case h
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type uT
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.Generators S T
      P : Algebra.Generators R S
      ⊢ Function.Injective ⇑Algebra.Extension.h1Cotangentι
    -/
  · exact Subtype.val_injective
    /-
      🎉 no goals
    -/


lemma δ_map
    (Q : Generators.{u₁} S T) (P : Generators.{u₂} R S)
    (Q' : Generators.{u₃} S T) (P' : Generators.{u₄} R S) (f : Hom Q' Q) (x) :
    δ Q P (Extension.H1Cotangent.map f.toExtensionHom x) = δ Q' P' x := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    Q' : Algebra.Generators S T
    P' : Algebra.Generators R S
    f : Q'.Hom Q
    x : Q'.toExtension.H1Cotangent
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δ Q P) ((Algebra.Extension.H1Cotangent.m …
  -/
  letI : AddCommGroup (T ⊗[S] Ω[S⁄R]) := inferInstance
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    Q' : Algebra.Generators S T
    P' : Algebra.Generators R S
    f : Q'.Hom Q
    x : Q'.toExtension.H1Cotangent
    this : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferInst …
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δ Q P) ((Algebra.Extension.H1Cotangent.m …
  -/
  obtain ⟨x, hx⟩ := x
  /-
    case mk
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    Q' : Algebra.Generators S T
    P' : Algebra.Generators R S
    f : Q'.Hom Q
    this : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferInst …
    x : Q'.toExtension.Cotangent
    hx : Membership.mem (LinearMap.ker Q'.toExtension.cotangentComplex) x
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δ Q P) ((Algebra.Extension.H1Cotangent.m …
  -/
  obtain ⟨⟨y, hy⟩, rfl⟩ := Extension.Cotangent.mk_surjective x
  /-
    case mk.intro.mk
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    Q' : Algebra.Generators S T
    P' : Algebra.Generators R S
    f : Q'.Hom Q
    this : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferInst …
    y : Q'.toExtension.Ring
    hy : Membership.mem Q'.toExtension.ker y
    hx : Membership.mem (LinearMap.ker Q'.toExtension.cotangentComplex) (Algebra.E …
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δ Q P) ((Algebra.Extension.H1Cotangent.m …
  -/
  show δ _ _ ⟨_, _⟩ = δ _ _ _
  replace hx : (1 : T) ⊗ₜ[Q'.Ring] (D S Q'.Ring) y = 0 := by
    simpa only [LinearMap.mem_ker, Extension.cotangentComplex_mk, ker, RingHom.mem_ker] using hx
  /-
    case mk.intro.mk
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    Q' : Algebra.Generators S T
    P' : Algebra.Generators R S
    f : Q'.Hom Q
    this : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferInst …
    y : Q'.toExtension.Ring
    hy : Membership.mem Q'.toExtension.ker y
    hx✝ : Membership.mem (LinearMap.ker Q'.toExtension.cotangentComplex) (Algebra. …
    hx : Eq (TensorProduct.tmul Q'.Ring 1 ((KaehlerDifferential.D S Q'.Ring) y)) 0
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δ Q P) ⟨((Algebra.Extension.Cotangent.ma …
  -/
  simp only [LinearMap.domRestrict_apply, Extension.Cotangent.map_mk, δ_eq_δAux]
  /-
    case mk.intro.mk
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    Q' : Algebra.Generators S T
    P' : Algebra.Generators R S
    f : Q'.Hom Q
    this : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferInst …
    y : Q'.toExtension.Ring
    hy : Membership.mem Q'.toExtension.ker y
    hx✝ : Membership.mem (LinearMap.ker Q'.toExtension.cotangentComplex) (Algebra. …
    hx : Eq (TensorProduct.tmul Q'.Ring 1 ((KaehlerDifferential.D S Q'.Ring) y)) 0
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δAux R Q) (f.toExtensionHom.toAlgHom y)) …
  -/
  refine (δAux_toAlgHom f _).trans ?_
  /-
    case mk.intro.mk
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    Q' : Algebra.Generators S T
    P' : Algebra.Generators R S
    f : Q'.Hom Q
    this : AddCommGroup (TensorProduct S T (KaehlerDifferential R S)) := inferInst …
    y : Q'.toExtension.Ring
    hy : Membership.mem Q'.toExtension.ker y
    hx✝ : Membership.mem (LinearMap.ker Q'.toExtension.cotangentComplex) (Algebra. …
    hx : Eq (TensorProduct.tmul Q'.Ring 1 ((KaehlerDifferential.D S Q'.Ring) y)) 0
    ⊢ Eq (HAdd.hAdd ((Algebra.Generators.H1Cotangent.δAux R Q') y) ((Finsupp.linea …
  -/
  rw [hx, map_zero, map_zero, add_zero]
  /-
    🎉 no goals
  -/


lemma δ_comp_equiv
    (Q : Generators.{u₁} S T) (P : Generators.{u₂} R S)
    (Q' : Generators.{u₃} S T) (P' : Generators.{u₄} R S) :
    δ Q P ∘ₗ (H1Cotangent.equiv _ _).toLinearMap = δ Q' P' := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    Q' : Algebra.Generators S T
    P' : Algebra.Generators R S
    ⊢ Eq ((Algebra.Generators.H1Cotangent.δ Q P).comp ↑(Algebra.Generators.H1Cotan …
  -/
  ext x
  /-
    case h
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    Q' : Algebra.Generators S T
    P' : Algebra.Generators R S
    x : Q'.toExtension.H1Cotangent
    ⊢ Eq (((Algebra.Generators.H1Cotangent.δ Q P).comp ↑(Algebra.Generators.H1Cota …
  -/
  exact δ_map Q P Q' P' _ _
  /-
    🎉 no goals
  -/


/-- A variant of `exact_map_δ` that takes in an arbitrary map between generators. -/
lemma exact_map_δ'
    (Q : Generators.{u₁} S T) (P : Generators.{u₂} R S) (P' : Generators.{u₃} R T) (f : Hom P' Q) :
    Function.Exact (Extension.H1Cotangent.map f.toExtensionHom) (δ Q P) := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    P' : Algebra.Generators R T
    f : P'.Hom Q
    ⊢ Function.Exact ⇑(Algebra.Extension.H1Cotangent.map f.toExtensionHom) ⇑(Algeb …
  -/
  refine (H1Cotangent.equiv (Q.comp P) P').surjective.comp_exact_iff_exact.mp ?_
  show Function.Exact ((Extension.H1Cotangent.map f.toExtensionHom).restrictScalars T ∘ₗ
    (Extension.H1Cotangent.map _)) (δ Q P)
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    P' : Algebra.Generators R T
    f : P'.Hom Q
    ⊢ Function.Exact ⇑((↑T (Algebra.Extension.H1Cotangent.map f.toExtensionHom)).c …
  -/
  rw [← Extension.H1Cotangent.map_comp, Extension.H1Cotangent.map_eq _ (Q.ofComp P).toExtensionHom]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type uT
    inst✝³ : CommRing T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    Q : Algebra.Generators S T
    P : Algebra.Generators R S
    P' : Algebra.Generators R T
    f : P'.Hom Q
    ⊢ Function.Exact ⇑(Algebra.Extension.H1Cotangent.map (Q.ofComp P).toExtensionH …
  -/
  exact exact_map_δ Q P
  /-
    🎉 no goals
  -/


/-- The connecting homomorphism in the Jacobi-Zariski sequence. -/
noncomputable
def H1Cotangent.δ : H1Cotangent S T →ₗ[T] T ⊗[S] Ω[S⁄R] :=
  Generators.H1Cotangent.δ (Generators.self S T) (Generators.self R S)


/-- Given algebras `R → S → T`, `H¹(L_{T/R}) → H¹(L_{T/S}) → T ⊗[S] Ω[S/R]` is exact. -/
lemma H1Cotangent.exact_map_δ : Function.Exact (map R S T T) (δ R S T) :=
  Generators.H1Cotangent.exact_map_δ' (Generators.self S T)
    (Generators.self R S) (Generators.self R T) (Generators.defaultHom _ _)


/-- Given algebras `R → S → T`, `H¹(L_{T/S}) → T ⊗[S] Ω[S/R] → Ω[T/R]` is exact. -/
lemma H1Cotangent.exact_δ_mapBaseChange : Function.Exact (δ R S T) (mapBaseChange R S T) :=
  Generators.H1Cotangent.exact_δ_map (Generators.self S T) (Generators.self R S)


