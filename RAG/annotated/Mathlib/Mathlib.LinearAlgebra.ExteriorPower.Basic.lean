/-- `exteriorAlgebra.ιMulti` is the alternating map from `Fin n → M` to `⋀[r]^n M`
induced by `exteriorAlgebra.ιMulti`, i.e. sending a family of vectors `m : Fin n → M` to the
product of its entries. -/
def ιMulti : M [⋀^Fin n]→ₗ[R] (⋀[R]^n M) :=
  (ExteriorAlgebra.ιMulti R n).codRestrict (⋀[R]^n M) fun _ =>
    ExteriorAlgebra.ιMulti_range R n <| Set.mem_range_self _


@[simp] lemma ιMulti_apply_coe (a : Fin n → M) : ιMulti R n a = ExteriorAlgebra.ιMulti R n a := rfl


/-- The image of `ExteriorAlgebra.ιMulti R n` spans the `n`th exterior power. Variant of
`ExteriorAlgebra.ιMulti_span_fixedDegree`, useful in rewrites. -/
lemma ιMulti_span_fixedDegree :
    Submodule.span R (Set.range (ExteriorAlgebra.ιMulti R n)) = ⋀[R]^n M :=
  ExteriorAlgebra.ιMulti_span_fixedDegree R n


/-- The image of `exteriorPower.ιMulti` spans `⋀[R]^n M`. -/
lemma ιMulti_span :
    Submodule.span R (Set.range (ιMulti R n)) = (⊤ : Submodule R (⋀[R]^n M)) := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Nat
    M : Type u_1
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Eq (Submodule.span R (Set.range ⇑(exteriorPower.ιMulti R n))) Top.top
  -/
  apply LinearMap.map_injective (Submodule.ker_subtype (⋀[R]^n M))
  /-
    case a
    R : Type u
    inst✝² : CommRing R
    n : Nat
    M : Type u_1
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Eq (Submodule.map (ExteriorAlgebra.exteriorPower R n M).subtype (Submodule.s …
  -/
  rw [LinearMap.map_span, ← Set.image_univ, Set.image_image]
  simp only [Submodule.coe_subtype, ιMulti_apply_coe, Set.image_univ, Submodule.map_top,
    Submodule.range_subtype]
  /-
    case a
    R : Type u
    inst✝² : CommRing R
    n : Nat
    M : Type u_1
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Eq (Submodule.span R (Set.range fun a => (ExteriorAlgebra.ιMulti R n) a)) (E …
  -/
  exact ExteriorAlgebra.ιMulti_span_fixedDegree R n
  /-
    🎉 no goals
  -/


/-- The index type for the relations in the standard presentation of `⋀[R]^n M`,
in the particular case `ι` is `Fin n`. -/
inductive Rels (ι : Type*) (M : Type*)
  | add (m : ι → M) (i : ι) (x y : M)
  | smul (m : ι → M) (i : ι) (r : R) (x : M)
  | alt (m : ι → M) (i j : ι) (hm : m i = m j) (hij : i ≠ j)


/-- The relations in the standard presentation of `⋀[R]^n M` with generators and relations. -/
@[simps]
noncomputable def relations (ι : Type*) [DecidableEq ι] (M : Type*)
    [AddCommGroup M] [Module R M] :
    Module.Relations R where
  G := ι → M
  R := Rels R ι M
  relation r := match r with
    | .add m i x y => Finsupp.single (update m i x) 1 +
        Finsupp.single (update m i y) 1 -
        Finsupp.single (update m i (x + y)) 1
    | .smul m i r x => Finsupp.single (update m i (r • x)) 1 -
        r • Finsupp.single (update m i x) 1
    | .alt m _ _ _ _ => Finsupp.single m 1


variable {R} in
/-- The solutions in a module `N` to the linear equations
given by `exteriorPower.relations R ι M` identify to alternating maps to `N`. -/
@[simps!]
def relationsSolutionEquiv {ι : Type*} [DecidableEq ι] {M : Type*}
    [AddCommGroup M] [Module R M] :
    (relations R ι M).Solution N ≃ AlternatingMap R M N ι where
  toFun s :=
    { toFun := fun m ↦ s.var m
      map_update_add' := fun m i x y ↦ by
        /-
          R : Type u
          inst✝¹⁰ : CommRing R
          n : Nat
          M✝ : Type u_1
          N : Type u_2
          N' : Type u_3
          inst✝⁹ : AddCommGroup M✝
          inst✝⁸ : Module R M✝
          inst✝⁷ : AddCommGroup N
          inst✝⁶ : Module R N
          inst✝⁵ : AddCommGroup N'
          inst✝⁴ : Module R N'
          ι : Type u_4
          inst✝³ : DecidableEq ι
          M : Type u_5
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          s : (exteriorPower.presentation.relations R ι M).Solution N
          inst✝ : DecidableEq ι
          m : ι → M
          i : ι
          x y : M
          ⊢ Eq ((fun m => s.var m) (Function.update m i (HAdd.hAdd x y))) (HAdd.hAdd ((f …
        -/
        have := s.linearCombination_var_relation (.add m i x y)
        /-
          R : Type u
          inst✝¹⁰ : CommRing R
          n : Nat
          M✝ : Type u_1
          N : Type u_2
          N' : Type u_3
          inst✝⁹ : AddCommGroup M✝
          inst✝⁸ : Module R M✝
          inst✝⁷ : AddCommGroup N
          inst✝⁶ : Module R N
          inst✝⁵ : AddCommGroup N'
          inst✝⁴ : Module R N'
          ι : Type u_4
          inst✝³ : DecidableEq ι
          M : Type u_5
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          s : (exteriorPower.presentation.relations R ι M).Solution N
          inst✝ : DecidableEq ι
          m : ι → M
          i : ι
          x y : M
          this : Eq ((Finsupp.linearCombination R s.var) ((exteriorPower.presentation.re …
          ⊢ Eq ((fun m => s.var m) (Function.update m i (HAdd.hAdd x y))) (HAdd.hAdd ((f …
        -/
        dsimp at this ⊢
        rw [map_sub, map_add, Finsupp.linearCombination_single, one_smul,
          Finsupp.linearCombination_single, one_smul,
          Finsupp.linearCombination_single, one_smul, sub_eq_zero] at this
        /-
          R : Type u
          inst✝¹⁰ : CommRing R
          n : Nat
          M✝ : Type u_1
          N : Type u_2
          N' : Type u_3
          inst✝⁹ : AddCommGroup M✝
          inst✝⁸ : Module R M✝
          inst✝⁷ : AddCommGroup N
          inst✝⁶ : Module R N
          inst✝⁵ : AddCommGroup N'
          inst✝⁴ : Module R N'
          ι : Type u_4
          inst✝³ : DecidableEq ι
          M : Type u_5
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          s : (exteriorPower.presentation.relations R ι M).Solution N
          inst✝ : DecidableEq ι
          m : ι → M
          i : ι
          x y : M
          this : Eq (HAdd.hAdd (s.var (Function.update m i x)) (s.var (Function.update m …
          ⊢ Eq (s.var (Function.update m i (HAdd.hAdd x y))) (HAdd.hAdd (s.var (Function …
        -/
        convert this.symm -- `convert` is necessary due to the implementation of `MultilinearMap`
        /-
          🎉 no goals
        -/
      map_update_smul' := fun m i r x ↦ by
        /-
          R : Type u
          inst✝¹⁰ : CommRing R
          n : Nat
          M✝ : Type u_1
          N : Type u_2
          N' : Type u_3
          inst✝⁹ : AddCommGroup M✝
          inst✝⁸ : Module R M✝
          inst✝⁷ : AddCommGroup N
          inst✝⁶ : Module R N
          inst✝⁵ : AddCommGroup N'
          inst✝⁴ : Module R N'
          ι : Type u_4
          inst✝³ : DecidableEq ι
          M : Type u_5
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          s : (exteriorPower.presentation.relations R ι M).Solution N
          inst✝ : DecidableEq ι
          m : ι → M
          i : ι
          r : R
          x : M
          ⊢ Eq ((fun m => s.var m) (Function.update m i (HSMul.hSMul r x))) (HSMul.hSMul …
        -/
        have := s.linearCombination_var_relation (.smul m i r x)
        /-
          R : Type u
          inst✝¹⁰ : CommRing R
          n : Nat
          M✝ : Type u_1
          N : Type u_2
          N' : Type u_3
          inst✝⁹ : AddCommGroup M✝
          inst✝⁸ : Module R M✝
          inst✝⁷ : AddCommGroup N
          inst✝⁶ : Module R N
          inst✝⁵ : AddCommGroup N'
          inst✝⁴ : Module R N'
          ι : Type u_4
          inst✝³ : DecidableEq ι
          M : Type u_5
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          s : (exteriorPower.presentation.relations R ι M).Solution N
          inst✝ : DecidableEq ι
          m : ι → M
          i : ι
          r : R
          x : M
          this : Eq ((Finsupp.linearCombination R s.var) ((exteriorPower.presentation.re …
          ⊢ Eq ((fun m => s.var m) (Function.update m i (HSMul.hSMul r x))) (HSMul.hSMul …
        -/
        dsimp at this ⊢
        rw [Finsupp.smul_single, smul_eq_mul, mul_one, map_sub,
          Finsupp.linearCombination_single, one_smul,
          Finsupp.linearCombination_single, sub_eq_zero] at this
        /-
          R : Type u
          inst✝¹⁰ : CommRing R
          n : Nat
          M✝ : Type u_1
          N : Type u_2
          N' : Type u_3
          inst✝⁹ : AddCommGroup M✝
          inst✝⁸ : Module R M✝
          inst✝⁷ : AddCommGroup N
          inst✝⁶ : Module R N
          inst✝⁵ : AddCommGroup N'
          inst✝⁴ : Module R N'
          ι : Type u_4
          inst✝³ : DecidableEq ι
          M : Type u_5
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          s : (exteriorPower.presentation.relations R ι M).Solution N
          inst✝ : DecidableEq ι
          m : ι → M
          i : ι
          r : R
          x : M
          this : Eq (s.var (Function.update m i (HSMul.hSMul r x))) (HSMul.hSMul r (s.va …
          ⊢ Eq (s.var (Function.update m i (HSMul.hSMul r x))) (HSMul.hSMul r (s.var (Fu …
        -/
        convert this
        /-
          🎉 no goals
        -/
      map_eq_zero_of_eq' := fun v i j hm hij ↦
           /-
             R : Type u
             inst✝⁹ : CommRing R
             n : Nat
             M✝ : Type u_1
             N : Type u_2
             N' : Type u_3
             inst✝⁸ : AddCommGroup M✝
             inst✝⁷ : Module R M✝
             inst✝⁶ : AddCommGroup N
             inst✝⁵ : Module R N
             inst✝⁴ : AddCommGroup N'
             inst✝³ : Module R N'
             ι : Type u_4
             inst✝² : DecidableEq ι
             M : Type u_5
             inst✝¹ : AddCommGroup M
             inst✝ : Module R M
             s : (exteriorPower.presentation.relations R ι M).Solution N
             v : ι → M
             i j : ι
             hm : Eq (v i) (v j)
             hij : Ne i j
             ⊢ Eq ({ toFun := fun m => s.var m, map_update_add' := ⋯, map_update_smul' := ⋯ …
           -/
        by simpa using s.linearCombination_var_relation (.alt v i j hm hij) }
           /-
             🎉 no goals
           -/
  invFun f :=
    { var := fun m ↦ f m
      linearCombination_var_relation := by
        /-
          R : Type u
          inst✝⁹ : CommRing R
          n : Nat
          M✝ : Type u_1
          N : Type u_2
          N' : Type u_3
          inst✝⁸ : AddCommGroup M✝
          inst✝⁷ : Module R M✝
          inst✝⁶ : AddCommGroup N
          inst✝⁵ : Module R N
          inst✝⁴ : AddCommGroup N'
          inst✝³ : Module R N'
          ι : Type u_4
          inst✝² : DecidableEq ι
          M : Type u_5
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          f : AlternatingMap R M N ι
          ⊢ ∀ (r : (exteriorPower.presentation.relations R ι M).R), Eq ((Finsupp.linearC …
        -/
        rintro (⟨m, i, x, y⟩ | ⟨m, i, r, x⟩ | ⟨v, i, j, hm, hij⟩)
          /-
            case add
            R : Type u
            inst✝⁹ : CommRing R
            n : Nat
            M✝ : Type u_1
            N : Type u_2
            N' : Type u_3
            inst✝⁸ : AddCommGroup M✝
            inst✝⁷ : Module R M✝
            inst✝⁶ : AddCommGroup N
            inst✝⁵ : Module R N
            inst✝⁴ : AddCommGroup N'
            inst✝³ : Module R N'
            ι : Type u_4
            inst✝² : DecidableEq ι
            M : Type u_5
            inst✝¹ : AddCommGroup M
            inst✝ : Module R M
            f : AlternatingMap R M N ι
            m : ι → M
            i : ι
            x y : M
            ⊢ Eq ((Finsupp.linearCombination R fun m => f m) ((exteriorPower.presentation. …
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case smul
            R : Type u
            inst✝⁹ : CommRing R
            n : Nat
            M✝ : Type u_1
            N : Type u_2
            N' : Type u_3
            inst✝⁸ : AddCommGroup M✝
            inst✝⁷ : Module R M✝
            inst✝⁶ : AddCommGroup N
            inst✝⁵ : Module R N
            inst✝⁴ : AddCommGroup N'
            inst✝³ : Module R N'
            ι : Type u_4
            inst✝² : DecidableEq ι
            M : Type u_5
            inst✝¹ : AddCommGroup M
            inst✝ : Module R M
            f : AlternatingMap R M N ι
            m : ι → M
            i : ι
            r : R
            x : M
            ⊢ Eq ((Finsupp.linearCombination R fun m => f m) ((exteriorPower.presentation. …
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case alt
            R : Type u
            inst✝⁹ : CommRing R
            n : Nat
            M✝ : Type u_1
            N : Type u_2
            N' : Type u_3
            inst✝⁸ : AddCommGroup M✝
            inst✝⁷ : Module R M✝
            inst✝⁶ : AddCommGroup N
            inst✝⁵ : Module R N
            inst✝⁴ : AddCommGroup N'
            inst✝³ : Module R N'
            ι : Type u_4
            inst✝² : DecidableEq ι
            M : Type u_5
            inst✝¹ : AddCommGroup M
            inst✝ : Module R M
            f : AlternatingMap R M N ι
            v : ι → M
            i j : ι
            hm : Eq (v i) (v j)
            hij : Ne i j
            ⊢ Eq ((Finsupp.linearCombination R fun m => f m) ((exteriorPower.presentation. …
          -/
        · simpa using f.map_eq_zero_of_eq v hm hij }
          /-
            🎉 no goals
          -/
  left_inv _ := rfl
  right_inv _ := rfl


/-- The universal property of the exterior power. -/
def isPresentationCore :
    (relationsSolutionEquiv.symm (ιMulti R n (M := M))).IsPresentationCore where
  desc s := LinearMap.comp (ExteriorAlgebra.liftAlternating
      (Function.update 0 n (relationsSolutionEquiv s))) (Submodule.subtype _)
                        /-
                          R : Type u
                          inst✝⁸ : CommRing R
                          n : Nat
                          M : Type u_1
                          N : Type u_2
                          N' : Type u_3
                          inst✝⁷ : AddCommGroup M
                          inst✝⁶ : Module R M
                          inst✝⁵ : AddCommGroup N
                          inst✝⁴ : Module R N
                          inst✝³ : AddCommGroup N'
                          inst✝² : Module R N'
                          N✝ : Type ?u.99734
                          inst✝¹ : AddCommGroup N✝
                          inst✝ : Module R N✝
                          s : (exteriorPower.presentation.relations R (Fin n) M).Solution N✝
                          ⊢ Eq ((exteriorPower.presentation.relationsSolutionEquiv.symm (exteriorPower.ι …
                        -/
  postcomp_desc s := by aesop
                        /-
                          🎉 no goals
                        -/
  postcomp_injective {N _ _ f f' h} := by
    /-
      R : Type u
      inst✝⁶ : CommRing R
      n : Nat
      M : Type u_1
      N✝ : Type u_2
      N' : Type u_3
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : AddCommGroup N✝
      inst✝² : Module R N✝
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      N : Type ?u.99734
      x✝¹ : AddCommGroup N
      x✝ : Module R N
      f f' : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (ExteriorAlge …
      h : Eq ((exteriorPower.presentation.relationsSolutionEquiv.symm (exteriorPower …
      ⊢ Eq f f'
    -/
    rw [Submodule.linearMap_eq_iff_of_span_eq_top _ _ (ιMulti_span R n M)]
    /-
      R : Type u
      inst✝⁶ : CommRing R
      n : Nat
      M : Type u_1
      N✝ : Type u_2
      N' : Type u_3
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : AddCommGroup N✝
      inst✝² : Module R N✝
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      N : Type ?u.99734
      x✝¹ : AddCommGroup N
      x✝ : Module R N
      f f' : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (ExteriorAlge …
      h : Eq ((exteriorPower.presentation.relationsSolutionEquiv.symm (exteriorPower …
      ⊢ ∀ (s : ↑(Set.range ⇑(exteriorPower.ιMulti R n))), Eq (f ↑s) (f' ↑s)
    -/
    rintro ⟨_, ⟨f, rfl⟩⟩
    /-
      case mk.intro
      R : Type u
      inst✝⁶ : CommRing R
      n : Nat
      M : Type u_1
      N✝ : Type u_2
      N' : Type u_3
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : AddCommGroup N✝
      inst✝² : Module R N✝
      inst✝¹ : AddCommGroup N'
      inst✝ : Module R N'
      N : Type ?u.99734
      x✝¹ : AddCommGroup N
      x✝ : Module R N
      f✝ f' : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (ExteriorAlg …
      h : Eq ((exteriorPower.presentation.relationsSolutionEquiv.symm (exteriorPower …
      f : Fin n → M
      ⊢ Eq (f✝ ↑⟨(exteriorPower.ιMulti R n) f, ⋯⟩) (f' ↑⟨(exteriorPower.ιMulti R n)  …
    -/
    exact Module.Relations.Solution.congr_var h f
    /-
      🎉 no goals
    -/


/-- The standard presentation of the `R`-module `⋀[R]^n M`. -/
@[simps! G R relation var]
noncomputable def presentation : Module.Presentation R (⋀[R]^n M) :=
  .ofIsPresentation (presentation.isPresentationCore R n M).isPresentation


/-- Two linear maps on `⋀[R]^n M` that agree on the image of `exteriorPower.ιMulti`
are equal. -/
@[ext]
lemma linearMap_ext {f : ⋀[R]^n M →ₗ[R] N} {g : ⋀[R]^n M →ₗ[R] N}
    (heq : f.compAlternatingMap (ιMulti R n) = g.compAlternatingMap (ιMulti R n)) : f = g :=
                                              /-
                                                R : Type u
                                                inst✝⁴ : CommRing R
                                                n : Nat
                                                M : Type u_1
                                                N : Type u_2
                                                inst✝³ : AddCommGroup M
                                                inst✝² : Module R M
                                                inst✝¹ : AddCommGroup N
                                                inst✝ : Module R N
                                                f g : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (ExteriorAlgeb …
                                                heq : Eq (f.compAlternatingMap (exteriorPower.ιMulti R n)) (g.compAlternatingM …
                                                ⊢ Eq ((exteriorPower.presentation R n M).postcomp f) ((exteriorPower.presentat …
                                              -/
  (presentation R n M).postcomp_injective (by ext f; apply DFunLike.congr_fun heq )
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- The linear equivalence between `n`-fold alternating maps from `M` to `N` and linear maps from
`⋀[R]^n M` to `N`: this is the universal property of the `n`th exterior power of `M`. -/
noncomputable def alternatingMapLinearEquiv : (M [⋀^Fin n]→ₗ[R] N) ≃ₗ[R] ⋀[R]^n M →ₗ[R] N :=
  LinearEquiv.symm
    (Equiv.toLinearEquiv
      ((presentation R n M).linearMapEquiv.trans presentation.relationsSolutionEquiv)
      { map_add := fun _ _ => rfl
        map_smul := fun _ _ => rfl })


@[simp]
lemma alternatingMapLinearEquiv_comp_ιMulti (f : M [⋀^Fin n]→ₗ[R] N) :
    (alternatingMapLinearEquiv f).compAlternatingMap (ιMulti R n) = f := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    n : Nat
    M : Type u_1
    N : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : AlternatingMap R M N (Fin n)
    ⊢ Eq ((exteriorPower.alternatingMapLinearEquiv f).compAlternatingMap (exterior …
  -/
  obtain ⟨φ, rfl⟩ := alternatingMapLinearEquiv.symm.surjective f
  /-
    case intro
    R : Type u
    inst✝⁴ : CommRing R
    n : Nat
    M : Type u_1
    N : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    φ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (ExteriorAlgebra …
    ⊢ Eq ((exteriorPower.alternatingMapLinearEquiv (exteriorPower.alternatingMapLi …
  -/
  dsimp [alternatingMapLinearEquiv]
  /-
    case intro
    R : Type u
    inst✝⁴ : CommRing R
    n : Nat
    M : Type u_1
    N : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    φ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (ExteriorAlgebra …
    ⊢ Eq ((((⋯.linearMapEquiv.trans exteriorPower.presentation.relationsSolutionEq …
  -/
  simp only [LinearEquiv.symm_apply_apply]
  /-
    case intro
    R : Type u
    inst✝⁴ : CommRing R
    n : Nat
    M : Type u_1
    N : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    φ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (ExteriorAlgebra …
    ⊢ Eq (φ.compAlternatingMap (exteriorPower.ιMulti R n)) (((⋯.linearMapEquiv.tra …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma alternatingMapLinearEquiv_apply_ιMulti (f : M [⋀^Fin n]→ₗ[R] N) (a : Fin n → M) :
    alternatingMapLinearEquiv f (ιMulti R n a) = f a :=
  DFunLike.congr_fun (alternatingMapLinearEquiv_comp_ιMulti f) a


@[simp]
lemma alternatingMapLinearEquiv_symm_apply (F : ⋀[R]^n M →ₗ[R] N) (m : Fin n → M) :
    alternatingMapLinearEquiv.symm F m = F.compAlternatingMap (ιMulti R n) m := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    n : Nat
    M : Type u_1
    N : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    F : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (ExteriorAlgebra …
    m : Fin n → M
    ⊢ Eq ((exteriorPower.alternatingMapLinearEquiv.symm F) m) ((F.compAlternatingM …
  -/
  obtain ⟨f, rfl⟩ := alternatingMapLinearEquiv.surjective F
  /-
    case intro
    R : Type u
    inst✝⁴ : CommRing R
    n : Nat
    M : Type u_1
    N : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    m : Fin n → M
    f : AlternatingMap R M N (Fin n)
    ⊢ Eq ((exteriorPower.alternatingMapLinearEquiv.symm (exteriorPower.alternating …
  -/
  simp only [LinearEquiv.symm_apply_apply, alternatingMapLinearEquiv_comp_ιMulti]
  /-
    🎉 no goals
  -/


@[simp]
lemma alternatingMapLinearEquiv_ιMulti :
    alternatingMapLinearEquiv (ιMulti R n (M := M)) = LinearMap.id := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Nat
    M : Type u_1
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Eq (exteriorPower.alternatingMapLinearEquiv (exteriorPower.ιMulti R n)) Line …
  -/
  ext
  simp only [alternatingMapLinearEquiv_comp_ιMulti, ιMulti_apply_coe,
    LinearMap.compAlternatingMap_apply, LinearMap.id_coe, id_eq]


/-- If `f` is an alternating map from `M` to `N`,
`alternatingMapLinearEquiv f` is the corresponding linear map from `⋀[R]^n M` to `N`,
and if `g` is a linear map from `N` to `N'`, then
the alternating map `g.compAlternatingMap f` from `M` to `N'` corresponds to the linear
map `g.comp (alternatingMapLinearEquiv f)` on `⋀[R]^n M`. -/
lemma alternatingMapLinearEquiv_comp (g : N →ₗ[R] N') (f : M [⋀^Fin n]→ₗ[R] N) :
    alternatingMapLinearEquiv (g.compAlternatingMap f) = g.comp (alternatingMapLinearEquiv f) := by
  /-
    R : Type u
    inst✝⁶ : CommRing R
    n : Nat
    M : Type u_1
    N : Type u_2
    N' : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : AddCommGroup N'
    inst✝ : Module R N'
    g : LinearMap (RingHom.id R) N N'
    f : AlternatingMap R M N (Fin n)
    ⊢ Eq (exteriorPower.alternatingMapLinearEquiv (g.compAlternatingMap f)) (g.com …
  -/
  ext
  simp only [alternatingMapLinearEquiv_comp_ιMulti, LinearMap.compAlternatingMap_apply,
    LinearMap.coe_comp, comp_apply, alternatingMapLinearEquiv_apply_ιMulti]


