/-- The 0th object in the complex of inhomogeneous cochains of `A : Rep k G` is isomorphic
to `A` as a `k`-module. -/
def zeroCochainsLequiv : (inhomogeneousCochains A).X 0 ≃ₗ[k] A :=
  LinearEquiv.funUnique (Fin 0 → G) k A


/-- The 1st object in the complex of inhomogeneous cochains of `A : Rep k G` is isomorphic
to `Fun(G, A)` as a `k`-module. -/
def oneCochainsLequiv : (inhomogeneousCochains A).X 1 ≃ₗ[k] G → A :=
  LinearEquiv.funCongrLeft k A (Equiv.funUnique (Fin 1) G).symm


/-- The 2nd object in the complex of inhomogeneous cochains of `A : Rep k G` is isomorphic
to `Fun(G², A)` as a `k`-module. -/
def twoCochainsLequiv : (inhomogeneousCochains A).X 2 ≃ₗ[k] G × G → A :=
  LinearEquiv.funCongrLeft k A <| (piFinTwoEquiv fun _ => G).symm


/-- The 3rd object in the complex of inhomogeneous cochains of `A : Rep k G` is isomorphic
to `Fun(G³, A)` as a `k`-module. -/
def threeCochainsLequiv : (inhomogeneousCochains A).X 3 ≃ₗ[k] G × G × G → A :=
  LinearEquiv.funCongrLeft k A <| ((Fin.consEquiv _).symm.trans
    ((Equiv.refl G).prodCongr (piFinTwoEquiv fun _ => G))).symm


/-- The 0th differential in the complex of inhomogeneous cochains of `A : Rep k G`, as a
`k`-linear map `A → Fun(G, A)`. It sends `(a, g) ↦ ρ_A(g)(a) - a.` -/
@[simps]
def dZero : A →ₗ[k] G → A where
  toFun m g := A.ρ g m - m
                                     /-
                                       k G : Type u
                                       inst✝¹ : CommRing k
                                       inst✝ : Group G
                                       A : Rep k G
                                       x y : CoeSort.coe A
                                       g : G
                                       ⊢ Eq ((fun m g => HSub.hSub ((A.ρ g) m) m) (HAdd.hAdd x y) g) (HAdd.hAdd ((fun …
                                     -/
  map_add' x y := funext fun g => by simp only [map_add, add_sub_add_comm]; rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                      /-
                                        k G : Type u
                                        inst✝¹ : CommRing k
                                        inst✝ : Group G
                                        A : Rep k G
                                        r : k
                                        x : CoeSort.coe A
                                        g : G
                                        ⊢ Eq ({ toFun := fun m g => HSub.hSub ((A.ρ g) m) m, map_add' := ⋯ }.toFun (HS …
                                      -/
  map_smul' r x := funext fun g => by dsimp; rw [map_smul, smul_sub]
                                             /-
                                               🎉 no goals
                                             -/


theorem dZero_ker_eq_invariants : LinearMap.ker (dZero A) = invariants A.ρ := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (LinearMap.ker (groupCohomology.dZero A)) A.ρ.invariants
  -/
  ext x
  /-
    case h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : CoeSort.coe A
    ⊢ Iff (Membership.mem (LinearMap.ker (groupCohomology.dZero A)) x) (Membership …
  -/
  simp only [LinearMap.mem_ker, mem_invariants, ← @sub_eq_zero _ _ _ x, funext_iff]
  /-
    case h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : CoeSort.coe A
    ⊢ Iff (∀ (x_1 : G), Eq ((groupCohomology.dZero A) x x_1) (0 x_1)) (∀ (g : G),  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] theorem dZero_eq_zero [A.IsTrivial] : dZero A = 0 := by
  /-
    k G : Type u
    inst✝² : CommRing k
    inst✝¹ : Group G
    A : Rep k G
    inst✝ : A.IsTrivial
    ⊢ Eq (groupCohomology.dZero A) 0
  -/
  ext
  /-
    case h.h
    k G : Type u
    inst✝² : CommRing k
    inst✝¹ : Group G
    A : Rep k G
    inst✝ : A.IsTrivial
    x✝¹ : CoeSort.coe A
    x✝ : G
    ⊢ Eq ((groupCohomology.dZero A) x✝¹ x✝) (0 x✝¹ x✝)
  -/
  simp only [dZero_apply, apply_eq_self, sub_self, LinearMap.zero_apply, Pi.zero_apply]
  /-
    🎉 no goals
  -/


/-- The 1st differential in the complex of inhomogeneous cochains of `A : Rep k G`, as a
`k`-linear map `Fun(G, A) → Fun(G × G, A)`. It sends
`(f, (g₁, g₂)) ↦ ρ_A(g₁)(f(g₂)) - f(g₁g₂) + f(g₁).` -/
@[simps]
def dOne : (G → A) →ₗ[k] G × G → A where
  toFun f g := A.ρ g.1 (f g.2) - f (g.1 * g.2) + f g.1
                                     /-
                                       k G : Type u
                                       inst✝¹ : CommRing k
                                       inst✝ : Group G
                                       A : Rep k G
                                       x y : G → CoeSort.coe A
                                       g : Prod G G
                                       ⊢ Eq ((fun f g => HAdd.hAdd (HSub.hSub ((A.ρ g.1) (f g.2)) (f (HMul.hMul g.1 g …
                                     -/
  map_add' x y := funext fun g => by dsimp; rw [map_add, add_add_add_comm, add_sub_add_comm]
                                            /-
                                              🎉 no goals
                                            -/
                                      /-
                                        k G : Type u
                                        inst✝¹ : CommRing k
                                        inst✝ : Group G
                                        A : Rep k G
                                        r : k
                                        x : G → CoeSort.coe A
                                        g : Prod G G
                                        ⊢ Eq ({ toFun := fun f g => HAdd.hAdd (HSub.hSub ((A.ρ g.1) (f g.2)) (f (HMul. …
                                      -/
  map_smul' r x := funext fun g => by dsimp; rw [map_smul, smul_add, smul_sub]
                                             /-
                                               🎉 no goals
                                             -/


/-- The 2nd differential in the complex of inhomogeneous cochains of `A : Rep k G`, as a
`k`-linear map `Fun(G × G, A) → Fun(G × G × G, A)`. It sends
`(f, (g₁, g₂, g₃)) ↦ ρ_A(g₁)(f(g₂, g₃)) - f(g₁g₂, g₃) + f(g₁, g₂g₃) - f(g₁, g₂).` -/
@[simps]
def dTwo : (G × G → A) →ₗ[k] G × G × G → A where
  toFun f g :=
    A.ρ g.1 (f (g.2.1, g.2.2)) - f (g.1 * g.2.1, g.2.2) + f (g.1, g.2.1 * g.2.2) - f (g.1, g.2.1)
  map_add' x y :=
    funext fun g => by
      /-
        k G : Type u
        inst✝¹ : CommRing k
        inst✝ : Group G
        A : Rep k G
        x y : Prod G G → CoeSort.coe A
        g : Prod G (Prod G G)
        ⊢ Eq ((fun f g => HSub.hSub (HAdd.hAdd (HSub.hSub ((A.ρ g.1) (f { fst := g.2.1 …
      -/
      dsimp
      rw [map_add, add_sub_add_comm (A.ρ _ _), add_sub_assoc, add_sub_add_comm, add_add_add_comm,
        add_sub_assoc, add_sub_assoc]
                                      /-
                                        k G : Type u
                                        inst✝¹ : CommRing k
                                        inst✝ : Group G
                                        A : Rep k G
                                        r : k
                                        x : Prod G G → CoeSort.coe A
                                        g : Prod G (Prod G G)
                                        ⊢ Eq ({ toFun := fun f g => HSub.hSub (HAdd.hAdd (HSub.hSub ((A.ρ g.1) (f { fs …
                                      -/
  map_smul' r x := funext fun g => by dsimp; simp only [map_smul, smul_add, smul_sub]
                                             /-
                                               🎉 no goals
                                             -/


/-- Let `C(G, A)` denote the complex of inhomogeneous cochains of `A : Rep k G`. This lemma
says `dZero` gives a simpler expression for the 0th differential: that is, the following
square commutes:
```
  C⁰(G, A) ---d⁰---> C¹(G, A)
  |                    |
  |                    |
  |                    |
  v                    v
  A ---- dZero ---> Fun(G, A)
```
where the vertical arrows are `zeroCochainsLequiv` and `oneCochainsLequiv` respectively.
-/
theorem dZero_comp_eq : dZero A ∘ₗ (zeroCochainsLequiv A) =
    oneCochainsLequiv A ∘ₗ ((inhomogeneousCochains A).d 0 1).hom := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq ((groupCohomology.dZero A).comp ↑(groupCohomology.zeroCochainsLequiv A))  …
  -/
  ext x y
  /-
    case h.h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : ↑((groupCohomology.inhomogeneousCochains A).X 0)
    y : G
    ⊢ Eq (((groupCohomology.dZero A).comp ↑(groupCohomology.zeroCochainsLequiv A)) …
  -/
  show A.ρ y (x default) - x default = _ + ({0} : Finset _).sum _
  simp_rw [Fin.val_eq_zero, zero_add, pow_one, neg_smul, one_smul,
    Finset.sum_singleton, sub_eq_add_neg]
  /-
    case h.h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : ↑((groupCohomology.inhomogeneousCochains A).X 0)
    y : G
    ⊢ Eq (HAdd.hAdd ((A.ρ y) (x Inhabited.default)) (Neg.neg (x Inhabited.default) …
  -/
               /-
                 🎉 no goals
               -/
  rcongr i <;> exact Fin.elim0 i
               /-
                 🎉 no goals
               -/


/-- Let `C(G, A)` denote the complex of inhomogeneous cochains of `A : Rep k G`. This lemma
says `dOne` gives a simpler expression for the 1st differential: that is, the following
square commutes:
```
  C¹(G, A) ---d¹-----> C²(G, A)
    |                      |
    |                      |
    |                      |
    v                      v
  Fun(G, A) -dOne-> Fun(G × G, A)
```
where the vertical arrows are `oneCochainsLequiv` and `twoCochainsLequiv` respectively.
-/
theorem dOne_comp_eq : dOne A ∘ₗ oneCochainsLequiv A =
    twoCochainsLequiv A ∘ₗ ((inhomogeneousCochains A).d 1 2).hom := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq ((groupCohomology.dOne A).comp ↑(groupCohomology.oneCochainsLequiv A)) (( …
  -/
  ext x y
  /-
    case h.h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : ↑((groupCohomology.inhomogeneousCochains A).X 1)
    y : Prod G G
    ⊢ Eq (((groupCohomology.dOne A).comp ↑(groupCohomology.oneCochainsLequiv A)) x …
  -/
  show A.ρ y.1 (x _) - x _ + x _ =  _ + _
  /-
    case h.h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : ↑((groupCohomology.inhomogeneousCochains A).X 1)
    y : Prod G G
    ⊢ Eq (HAdd.hAdd (HSub.hSub ((A.ρ y.1) (x ((Equiv.funUnique (Fin 1) G).symm y.2 …
  -/
  rw [Fin.sum_univ_two]
  simp only [Fin.val_zero, zero_add, pow_one, neg_smul, one_smul, Fin.val_one,
    Nat.one_add, neg_one_sq, sub_eq_add_neg, add_assoc]
  /-
    case h.h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : ↑((groupCohomology.inhomogeneousCochains A).X 1)
    y : Prod G G
    ⊢ Eq (HAdd.hAdd ((A.ρ y.1) (x ((Equiv.funUnique (Fin 1) G).symm y.2))) (HAdd.h …
  -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
  rcongr i <;> rw [Subsingleton.elim i 0] <;> rfl
                                              /-
                                                🎉 no goals
                                              -/


/-- Let `C(G, A)` denote the complex of inhomogeneous cochains of `A : Rep k G`. This lemma
says `dTwo` gives a simpler expression for the 2nd differential: that is, the following
square commutes:
```
      C²(G, A) -------d²-----> C³(G, A)
        |                         |
        |                         |
        |                         |
        v                         v
  Fun(G × G, A) --dTwo--> Fun(G × G × G, A)
```
where the vertical arrows are `twoCochainsLequiv` and `threeCochainsLequiv` respectively.
-/
theorem dTwo_comp_eq :
    dTwo A ∘ₗ twoCochainsLequiv A =
      threeCochainsLequiv A ∘ₗ ((inhomogeneousCochains A).d 2 3).hom := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq ((groupCohomology.dTwo A).comp ↑(groupCohomology.twoCochainsLequiv A)) (( …
  -/
  ext x y
  /-
    case h.h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : ↑((groupCohomology.inhomogeneousCochains A).X 2)
    y : Prod G (Prod G G)
    ⊢ Eq (((groupCohomology.dTwo A).comp ↑(groupCohomology.twoCochainsLequiv A)) x …
  -/
  show A.ρ y.1 (x _) - x _ + x _ - x _ = _ + _
  /-
    case h.h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : ↑((groupCohomology.inhomogeneousCochains A).X 2)
    y : Prod G (Prod G G)
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub ((A.ρ y.1) (x ((piFinTwoEquiv fun x => G …
  -/
  dsimp
  /-
    case h.h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : ↑((groupCohomology.inhomogeneousCochains A).X 2)
    y : Prod G (Prod G G)
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub ((A.ρ y.1) (x (Fin.cons y.2.1 (Fin.cons  …
  -/
  rw [Fin.sum_univ_three]
  simp only [sub_eq_add_neg, add_assoc, Fin.val_zero, zero_add, pow_one, neg_smul,
    one_smul, Fin.val_one, Fin.val_two, pow_succ' (-1 : k) 2, neg_sq, Nat.one_add, one_pow, mul_one]
  /-
    case h.h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : ↑((groupCohomology.inhomogeneousCochains A).X 2)
    y : Prod G (Prod G G)
    ⊢ Eq (HAdd.hAdd ((A.ρ y.1) (x (Fin.cons y.2.1 (Fin.cons y.2.2 finZeroElim))))  …
  -/
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 🎉 no goals
                               -/
  rcongr i <;> fin_cases i <;> rfl
                               /-
                                 🎉 no goals
                               -/


theorem dOne_comp_dZero : dOne A ∘ₗ dZero A = 0 := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq ((groupCohomology.dOne A).comp (groupCohomology.dZero A)) 0
  -/
  ext x g
  simp only [LinearMap.coe_comp, Function.comp_apply, dOne_apply A, dZero_apply A, map_sub,
    map_mul, LinearMap.mul_apply, sub_sub_sub_cancel_left, sub_add_sub_cancel, sub_self]
  /-
    case h.h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : CoeSort.coe A
    g : Prod G G
    ⊢ Eq 0 (0 x g)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem dTwo_comp_dOne : dTwo A ∘ₗ dOne A = 0 := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq ((groupCohomology.dTwo A).comp (groupCohomology.dOne A)) 0
  -/
  show (ModuleCat.ofHom (dOne A) ≫ ModuleCat.ofHom (dTwo A)).hom = _
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom (groupCohomology.dOn …
  -/
  have h1 := congr_arg ModuleCat.ofHom (dOne_comp_eq A)
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    h1 : Eq (ModuleCat.ofHom ((groupCohomology.dOne A).comp ↑(groupCohomology.oneC …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom (groupCohomology.dOn …
  -/
  have h2 := congr_arg ModuleCat.ofHom (dTwo_comp_eq A)
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    h1 : Eq (ModuleCat.ofHom ((groupCohomology.dOne A).comp ↑(groupCohomology.oneC …
    h2 : Eq (ModuleCat.ofHom ((groupCohomology.dTwo A).comp ↑(groupCohomology.twoC …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom (groupCohomology.dOn …
  -/
  simp only [ModuleCat.ofHom_comp, ModuleCat.ofHom_comp, ← LinearEquiv.toModuleIso_hom_hom] at h1 h2
  simp only [(Iso.eq_inv_comp _).2 h2, (Iso.eq_inv_comp _).2 h1, ModuleCat.ofHom_hom,
    ModuleCat.hom_ofHom, Category.assoc, Iso.hom_inv_id_assoc, HomologicalComplex.d_comp_d_assoc,
    zero_comp, comp_zero, ModuleCat.hom_zero]


/-- The 1-cocycles `Z¹(G, A)` of `A : Rep k G`, defined as the kernel of the map
`Fun(G, A) → Fun(G × G, A)` sending `(f, (g₁, g₂)) ↦ ρ_A(g₁)(f(g₂)) - f(g₁g₂) + f(g₁).` -/
def oneCocycles : Submodule k (G → A) := LinearMap.ker (dOne A)


/-- The 2-cocycles `Z²(G, A)` of `A : Rep k G`, defined as the kernel of the map
`Fun(G × G, A) → Fun(G × G × G, A)` sending
`(f, (g₁, g₂, g₃)) ↦ ρ_A(g₁)(f(g₂, g₃)) - f(g₁g₂, g₃) + f(g₁, g₂g₃) - f(g₁, g₂).` -/
def twoCocycles : Submodule k (G × G → A) := LinearMap.ker (dTwo A)


instance : FunLike (oneCocycles A) G A := ⟨Subtype.val, Subtype.val_injective⟩


@[simp]
theorem oneCocycles.coe_mk (f : G → A) (hf) : ((⟨f, hf⟩ : oneCocycles A) : G → A) = f := rfl


@[simp]
theorem oneCocycles.val_eq_coe (f : oneCocycles A) : f.1 = f := rfl


@[ext]
theorem oneCocycles_ext {f₁ f₂ : oneCocycles A} (h : ∀ g : G, f₁ g = f₂ g) : f₁ = f₂ :=
  DFunLike.ext f₁ f₂ h


theorem mem_oneCocycles_def (f : G → A) :
    f ∈ oneCocycles A ↔ ∀ g h : G, A.ρ g (f h) - f (g * h) + f g = 0 :=
  LinearMap.mem_ker.trans <| by
    /-
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Group G
      A : Rep k G
      f : G → CoeSort.coe A
      ⊢ Iff (Eq ((groupCohomology.dOne A) f) 0) (∀ (g h : G), Eq (HAdd.hAdd (HSub.hS …
    -/
    rw [funext_iff]
    /-
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Group G
      A : Rep k G
      f : G → CoeSort.coe A
      ⊢ Iff (∀ (x : Prod G G), Eq ((groupCohomology.dOne A) f x) (0 x)) (∀ (g h : G) …
    -/
    simp only [dOne_apply, Pi.zero_apply, Prod.forall]
    /-
      🎉 no goals
    -/


theorem mem_oneCocycles_iff (f : G → A) :
    f ∈ oneCocycles A ↔ ∀ g h : G, f (g * h) = A.ρ g (f h) + f g := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    f : G → CoeSort.coe A
    ⊢ Iff (Membership.mem (groupCohomology.oneCocycles A) f) (∀ (g h : G), Eq (f ( …
  -/
  simp_rw [mem_oneCocycles_def, sub_add_eq_add_sub, sub_eq_zero, eq_comm]
  /-
    🎉 no goals
  -/


@[simp] theorem oneCocycles_map_one (f : oneCocycles A) : f 1 = 0 := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    f : Subtype fun x => Membership.mem (groupCohomology.oneCocycles A) x
    ⊢ Eq (f 1) 0
  -/
  have := (mem_oneCocycles_def f).1 f.2 1 1
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    f : Subtype fun x => Membership.mem (groupCohomology.oneCocycles A) x
    this : Eq (HAdd.hAdd (HSub.hSub ((A.ρ 1) (f 1)) (f (HMul.hMul 1 1))) (f 1)) 0
    ⊢ Eq (f 1) 0
  -/
  simpa only [map_one, LinearMap.one_apply, mul_one, sub_self, zero_add] using this
  /-
    🎉 no goals
  -/


@[simp] theorem oneCocycles_map_inv (f : oneCocycles A) (g : G) :
    A.ρ g (f g⁻¹) = - f g := by
  rw [← add_eq_zero_iff_eq_neg, ← oneCocycles_map_one f, ← mul_inv_cancel g,
    (mem_oneCocycles_iff f).1 f.2 g g⁻¹]


theorem oneCocycles_map_mul_of_isTrivial [A.IsTrivial] (f : oneCocycles A) (g h : G) :
    f (g * h) = f g + f h := by
  /-
    k G : Type u
    inst✝² : CommRing k
    inst✝¹ : Group G
    A : Rep k G
    inst✝ : A.IsTrivial
    f : Subtype fun x => Membership.mem (groupCohomology.oneCocycles A) x
    g h : G
    ⊢ Eq (f (HMul.hMul g h)) (HAdd.hAdd (f g) (f h))
  -/
  rw [(mem_oneCocycles_iff f).1 f.2, apply_eq_self A.ρ g (f h), add_comm]
  /-
    🎉 no goals
  -/


theorem mem_oneCocycles_of_addMonoidHom [A.IsTrivial] (f : Additive G →+ A) :
    f ∘ Additive.ofMul ∈ oneCocycles A :=
  (mem_oneCocycles_iff _).2 fun g h => by
    simp only [Function.comp_apply, ofMul_mul, map_add,
      oneCocycles_map_mul_of_isTrivial, apply_eq_self A.ρ g (f (Additive.ofMul h)),
      add_comm (f (Additive.ofMul g))]


/-- When `A : Rep k G` is a trivial representation of `G`, `Z¹(G, A)` is isomorphic to the
group homs `G → A`. -/
@[simps] def oneCocyclesLequivOfIsTrivial [hA : A.IsTrivial] :
    oneCocycles A ≃ₗ[k] Additive G →+ A where
  toFun f :=
    { toFun := f ∘ Additive.toMul
      map_zero' := oneCocycles_map_one f
      map_add' := oneCocycles_map_mul_of_isTrivial f }
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  invFun f :=
    { val := f
      property := mem_oneCocycles_of_addMonoidHom f }
                   /-
                     k G : Type u
                     inst✝¹ : CommRing k
                     inst✝ : Group G
                     A : Rep k G
                     hA : A.IsTrivial
                     f : Subtype fun x => Membership.mem (groupCohomology.oneCocycles A) x
                     ⊢ Eq ((fun f => ⟨⇑f, ⋯⟩) ({ toFun := fun f => { toFun := Function.comp ⇑f ⇑Add …
                   -/
  left_inv f := by ext; rfl
                        /-
                          🎉 no goals
                        -/
                    /-
                      k G : Type u
                      inst✝¹ : CommRing k
                      inst✝ : Group G
                      A : Rep k G
                      hA : A.IsTrivial
                      f : AddMonoidHom (Additive G) (CoeSort.coe A)
                      ⊢ Eq ({ toFun := fun f => { toFun := Function.comp ⇑f ⇑Additive.toMul, map_zer …
                    -/
  right_inv f := by ext; rfl
                         /-
                           🎉 no goals
                         -/


instance : FunLike (twoCocycles A) (G × G) A := ⟨Subtype.val, Subtype.val_injective⟩


@[simp]
theorem twoCocycles.coe_mk (f : G × G → A) (hf) : ((⟨f, hf⟩ : twoCocycles A) : G × G → A) = f := rfl


@[simp]
theorem twoCocycles.val_eq_coe (f : twoCocycles A) : f.1 = f := rfl


@[ext]
theorem twoCocycles_ext {f₁ f₂ : twoCocycles A} (h : ∀ g h : G, f₁ (g, h) = f₂ (g, h)) : f₁ = f₂ :=
  DFunLike.ext f₁ f₂ (Prod.forall.mpr h)


theorem mem_twoCocycles_def (f : G × G → A) :
    f ∈ twoCocycles A ↔ ∀ g h j : G,
      A.ρ g (f (h, j)) - f (g * h, j) + f (g, h * j) - f (g, h) = 0 :=
  LinearMap.mem_ker.trans <| by
    /-
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Group G
      A : Rep k G
      f : Prod G G → CoeSort.coe A
      ⊢ Iff (Eq ((groupCohomology.dTwo A) f) 0) (∀ (g h j : G), Eq (HSub.hSub (HAdd. …
    -/
    rw [funext_iff]
    /-
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Group G
      A : Rep k G
      f : Prod G G → CoeSort.coe A
      ⊢ Iff (∀ (x : Prod G (Prod G G)), Eq ((groupCohomology.dTwo A) f x) (0 x)) (∀  …
    -/
    simp only [dTwo_apply, Prod.mk.eta, Pi.zero_apply, Prod.forall]
    /-
      🎉 no goals
    -/


theorem mem_twoCocycles_iff (f : G × G → A) :
    f ∈ twoCocycles A ↔ ∀ g h j : G,
      f (g * h, j) + f (g, h) =
        A.ρ g (f (h, j)) + f (g, h * j) := by
  simp_rw [mem_twoCocycles_def, sub_eq_zero, sub_add_eq_add_sub, sub_eq_iff_eq_add, eq_comm,
    add_comm (f (_ * _, _))]


theorem twoCocycles_map_one_fst (f : twoCocycles A) (g : G) :
    f (1, g) = f (1, 1) := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    f : Subtype fun x => Membership.mem (groupCohomology.twoCocycles A) x
    g : G
    ⊢ Eq (f { fst := 1, snd := g }) (f { fst := 1, snd := 1 })
  -/
  have := ((mem_twoCocycles_iff f).1 f.2 1 1 g).symm
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    f : Subtype fun x => Membership.mem (groupCohomology.twoCocycles A) x
    g : G
    this : Eq (HAdd.hAdd ((A.ρ 1) (f { fst := 1, snd := g })) (f { fst := 1, snd : …
    ⊢ Eq (f { fst := 1, snd := g }) (f { fst := 1, snd := 1 })
  -/
  simpa only [map_one, LinearMap.one_apply, one_mul, add_right_inj, this]
  /-
    🎉 no goals
  -/


theorem twoCocycles_map_one_snd (f : twoCocycles A) (g : G) :
    f (g, 1) = A.ρ g (f (1, 1)) := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    f : Subtype fun x => Membership.mem (groupCohomology.twoCocycles A) x
    g : G
    ⊢ Eq (f { fst := g, snd := 1 }) ((A.ρ g) (f { fst := 1, snd := 1 }))
  -/
  have := (mem_twoCocycles_iff f).1 f.2 g 1 1
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    f : Subtype fun x => Membership.mem (groupCohomology.twoCocycles A) x
    g : G
    this : Eq (HAdd.hAdd (f { fst := HMul.hMul g 1, snd := 1 }) (f { fst := g, snd …
    ⊢ Eq (f { fst := g, snd := 1 }) ((A.ρ g) (f { fst := 1, snd := 1 }))
  -/
  simpa only [mul_one, add_left_inj, this]
  /-
    🎉 no goals
  -/


lemma twoCocycles_ρ_map_inv_sub_map_inv (f : twoCocycles A) (g : G) :
    A.ρ g (f (g⁻¹, g)) - f (g, g⁻¹)
      = f (1, 1) - f (g, 1) := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    f : Subtype fun x => Membership.mem (groupCohomology.twoCocycles A) x
    g : G
    ⊢ Eq (HSub.hSub ((A.ρ g) (f { fst := Inv.inv g, snd := g })) (f { fst := g, sn …
  -/
  have := (mem_twoCocycles_iff f).1 f.2 g g⁻¹ g
  simp only [mul_inv_cancel, inv_mul_cancel, twoCocycles_map_one_fst _ g]
    at this
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    f : Subtype fun x => Membership.mem (groupCohomology.twoCocycles A) x
    g : G
    this : Eq (HAdd.hAdd (f { fst := 1, snd := 1 }) (f { fst := g, snd := Inv.inv  …
    ⊢ Eq (HSub.hSub ((A.ρ g) (f { fst := Inv.inv g, snd := g })) (f { fst := g, sn …
  -/
  exact sub_eq_sub_iff_add_eq_add.2 this.symm
  /-
    🎉 no goals
  -/


/-- The 1-coboundaries `B¹(G, A)` of `A : Rep k G`, defined as the image of the map
`A → Fun(G, A)` sending `(a, g) ↦ ρ_A(g)(a) - a.` -/
def oneCoboundaries : Submodule k (oneCocycles A) :=
  LinearMap.range ((dZero A).codRestrict (oneCocycles A) fun c =>
    LinearMap.ext_iff.1 (dOne_comp_dZero A) c)


/-- The 2-coboundaries `B²(G, A)` of `A : Rep k G`, defined as the image of the map
`Fun(G, A) → Fun(G × G, A)` sending `(f, (g₁, g₂)) ↦ ρ_A(g₁)(f(g₂)) - f(g₁g₂) + f(g₁).` -/
def twoCoboundaries : Submodule k (twoCocycles A) :=
  LinearMap.range ((dOne A).codRestrict (twoCocycles A) fun c =>
    LinearMap.ext_iff.1 (dTwo_comp_dOne.{u} A) c)


/-- Makes a 1-coboundary out of `f ∈ Im(d⁰)`. -/
def oneCoboundariesOfMemRange {f : G → A} (h : f ∈ LinearMap.range (dZero A)) :
    oneCoboundaries A :=
  ⟨⟨f, LinearMap.range_le_ker_iff.2 (dOne_comp_dZero A) h⟩,
       /-
         k G : Type u
         inst✝¹ : CommRing k
         inst✝ : Group G
         A : Rep k G
         f : G → CoeSort.coe A
         h : Membership.mem (LinearMap.range (groupCohomology.dZero A)) f
         ⊢ Membership.mem (groupCohomology.oneCoboundaries A) ⟨f, ⋯⟩
       -/
    by rcases h with ⟨x, rfl⟩; exact ⟨x, rfl⟩⟩
                               /-
                                 🎉 no goals
                               -/


theorem oneCoboundaries_of_mem_range_apply {f : G → A} (h : f ∈ LinearMap.range (dZero A)) :
    (oneCoboundariesOfMemRange h).1.1 = f := rfl


/-- Makes a 1-coboundary out of `f : G → A` and `x` such that
`ρ(g)(x) - x = f(g)` for all `g : G`. -/
def oneCoboundariesOfEq {f : G → A} {x : A} (hf : ∀ g, A.ρ g x - x = f g) :
    oneCoboundaries A :=
  oneCoboundariesOfMemRange ⟨x, funext hf⟩


theorem oneCoboundariesOfEq_apply {f : G → A} {x : A} (hf : ∀ g, A.ρ g x - x = f g) :
    (oneCoboundariesOfEq hf).1.1 = f := rfl


theorem mem_range_of_mem_oneCoboundaries {f : oneCocycles A} (h : f ∈ oneCoboundaries A) :
    f.1 ∈ LinearMap.range (dZero A) := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    f : Subtype fun x => Membership.mem (groupCohomology.oneCocycles A) x
    h : Membership.mem (groupCohomology.oneCoboundaries A) f
    ⊢ Membership.mem (LinearMap.range (groupCohomology.dZero A)) ↑f
  -/
  rcases h with ⟨x, rfl⟩; exact ⟨x, rfl⟩
                          /-
                            🎉 no goals
                          -/


theorem oneCoboundaries_eq_bot_of_isTrivial (A : Rep k G) [A.IsTrivial] :
    oneCoboundaries A = ⊥ := by
  /-
    k G : Type u
    inst✝² : CommRing k
    inst✝¹ : Group G
    A : Rep k G
    inst✝ : A.IsTrivial
    ⊢ Eq (groupCohomology.oneCoboundaries A) Bot.bot
  -/
  simp_rw [oneCoboundaries, dZero_eq_zero]
  /-
    k G : Type u
    inst✝² : CommRing k
    inst✝¹ : Group G
    A : Rep k G
    inst✝ : A.IsTrivial
    ⊢ Eq (LinearMap.range (LinearMap.codRestrict (groupCohomology.oneCocycles A) 0 …
  -/
  exact LinearMap.range_eq_bot.2 rfl
  /-
    🎉 no goals
  -/


theorem mem_oneCoboundaries_iff (f : oneCocycles A) : f ∈ oneCoboundaries A ↔
    ∃ x : A, ∀ g : G, A.ρ g x - x = f g := exists_congr fun x ↦ by
  simpa only [LinearMap.codRestrict, dZero, LinearMap.coe_mk, AddHom.coe_mk] using
    groupCohomology.oneCocycles_ext_iff


/-- Makes a 2-coboundary out of `f ∈ Im(d¹)`. -/
def twoCoboundariesOfMemRange {f : G × G → A} (h : f ∈ LinearMap.range (dOne A)) :
    twoCoboundaries A :=
  ⟨⟨f, LinearMap.range_le_ker_iff.2 (dTwo_comp_dOne A) h⟩,
       /-
         k G : Type u
         inst✝¹ : CommRing k
         inst✝ : Group G
         A : Rep k G
         f : Prod G G → CoeSort.coe A
         h : Membership.mem (LinearMap.range (groupCohomology.dOne A)) f
         ⊢ Membership.mem (groupCohomology.twoCoboundaries A) ⟨f, ⋯⟩
       -/
    by rcases h with ⟨x, rfl⟩; exact ⟨x, rfl⟩⟩
                               /-
                                 🎉 no goals
                               -/


theorem twoCoboundariesOfMemRange_apply {f : G × G → A} (h : f ∈ LinearMap.range (dOne A)) :
    (twoCoboundariesOfMemRange h).1.1 = f := rfl


/-- Makes a 2-coboundary out of `f : G × G → A` and `x : G → A` such that
`ρ(g)(x(h)) - x(gh) + x(g) = f(g, h)` for all `g, h : G`. -/
def twoCoboundariesOfEq {f : G × G → A} {x : G → A}
    (hf : ∀ g h, A.ρ g (x h) - x (g * h) + x g = f (g, h)) :
    twoCoboundaries A :=
  twoCoboundariesOfMemRange ⟨x, funext fun g ↦ hf g.1 g.2⟩


theorem twoCoboundariesOfEq_apply {f : G × G → A} {x : G → A}
    (hf : ∀ g h, A.ρ g (x h) - x (g * h) + x g = f (g, h)) :
    (twoCoboundariesOfEq hf).1.1 = f := rfl


theorem mem_range_of_mem_twoCoboundaries {f : twoCocycles A} (h : f ∈ twoCoboundaries A) :
    (twoCocycles A).subtype f ∈ LinearMap.range (dOne A) := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    f : Subtype fun x => Membership.mem (groupCohomology.twoCocycles A) x
    h : Membership.mem (groupCohomology.twoCoboundaries A) f
    ⊢ Membership.mem (LinearMap.range (groupCohomology.dOne A)) ((groupCohomology. …
  -/
  rcases h with ⟨x, rfl⟩; exact ⟨x, rfl⟩
                          /-
                            🎉 no goals
                          -/


theorem mem_twoCoboundaries_iff (f : twoCocycles A) : f ∈ twoCoboundaries A ↔
    ∃ x : G → A, ∀ g h : G, A.ρ g (x h) - x (g * h) + x g = f (g, h) := exists_congr fun x ↦ by
  simpa only [LinearMap.codRestrict, dOne, LinearMap.coe_mk, AddHom.coe_mk] using
    groupCohomology.twoCocycles_ext_iff


/-- A function `f : G → A` satisfies the 1-cocycle condition if
`f(gh) = g • f(h) + f(g)` for all `g, h : G`. -/
def IsOneCocycle (f : G → A) : Prop := ∀ g h : G, f (g * h) = g • f h + f g


/-- A function `f : G × G → A` satisfies the 2-cocycle condition if
`f(gh, j) + f(g, h) = g • f(h, j) + f(g, hj)` for all `g, h : G`. -/
def IsTwoCocycle (f : G × G → A) : Prop :=
  ∀ g h j : G, f (g * h, j) + f (g, h) = g • (f (h, j)) + f (g, h * j)


theorem map_one_of_isOneCocycle {f : G → A} (hf : IsOneCocycle f) :
    f 1 = 0 := by
  /-
    G : Type u_1
    A : Type u_2
    inst✝² : Monoid G
    inst✝¹ : AddCommGroup A
    inst✝ : MulAction G A
    f : G → A
    hf : groupCohomology.IsOneCocycle f
    ⊢ Eq (f 1) 0
  -/
  simpa only [mul_one, one_smul, self_eq_add_right] using hf 1 1
  /-
    🎉 no goals
  -/


theorem map_one_fst_of_isTwoCocycle {f : G × G → A} (hf : IsTwoCocycle f) (g : G) :
    f (1, g) = f (1, 1) := by
  /-
    G : Type u_1
    A : Type u_2
    inst✝² : Monoid G
    inst✝¹ : AddCommGroup A
    inst✝ : MulAction G A
    f : Prod G G → A
    hf : groupCohomology.IsTwoCocycle f
    g : G
    ⊢ Eq (f { fst := 1, snd := g }) (f { fst := 1, snd := 1 })
  -/
  simpa only [one_smul, one_mul, mul_one, add_right_inj] using (hf 1 1 g).symm
  /-
    🎉 no goals
  -/


theorem map_one_snd_of_isTwoCocycle {f : G × G → A} (hf : IsTwoCocycle f) (g : G) :
    f (g, 1) = g • f (1, 1) := by
  /-
    G : Type u_1
    A : Type u_2
    inst✝² : Monoid G
    inst✝¹ : AddCommGroup A
    inst✝ : MulAction G A
    f : Prod G G → A
    hf : groupCohomology.IsTwoCocycle f
    g : G
    ⊢ Eq (f { fst := g, snd := 1 }) (HSMul.hSMul g (f { fst := 1, snd := 1 }))
  -/
  simpa only [mul_one, add_left_inj] using hf g 1 1
  /-
    🎉 no goals
  -/


@[scoped simp] theorem map_inv_of_isOneCocycle {f : G → A} (hf : IsOneCocycle f) (g : G) :
    g • f g⁻¹ = - f g := by
  /-
    G : Type u_1
    A : Type u_2
    inst✝² : Group G
    inst✝¹ : AddCommGroup A
    inst✝ : MulAction G A
    f : G → A
    hf : groupCohomology.IsOneCocycle f
    g : G
    ⊢ Eq (HSMul.hSMul g (f (Inv.inv g))) (Neg.neg (f g))
  -/
  rw [← add_eq_zero_iff_eq_neg, ← map_one_of_isOneCocycle hf, ← mul_inv_cancel g, hf g g⁻¹]
  /-
    🎉 no goals
  -/


theorem smul_map_inv_sub_map_inv_of_isTwoCocycle {f : G × G → A} (hf : IsTwoCocycle f) (g : G) :
    g • f (g⁻¹, g) - f (g, g⁻¹) = f (1, 1) - f (g, 1) := by
  /-
    G : Type u_1
    A : Type u_2
    inst✝² : Group G
    inst✝¹ : AddCommGroup A
    inst✝ : MulAction G A
    f : Prod G G → A
    hf : groupCohomology.IsTwoCocycle f
    g : G
    ⊢ Eq (HSub.hSub (HSMul.hSMul g (f { fst := Inv.inv g, snd := g })) (f { fst := …
  -/
  have := hf g g⁻¹ g
  /-
    G : Type u_1
    A : Type u_2
    inst✝² : Group G
    inst✝¹ : AddCommGroup A
    inst✝ : MulAction G A
    f : Prod G G → A
    hf : groupCohomology.IsTwoCocycle f
    g : G
    this : Eq (HAdd.hAdd (f { fst := HMul.hMul g (Inv.inv g), snd := g }) (f { fst …
    ⊢ Eq (HSub.hSub (HSMul.hSMul g (f { fst := Inv.inv g, snd := g })) (f { fst := …
  -/
  simp only [mul_inv_cancel, inv_mul_cancel, map_one_fst_of_isTwoCocycle hf g] at this
  /-
    G : Type u_1
    A : Type u_2
    inst✝² : Group G
    inst✝¹ : AddCommGroup A
    inst✝ : MulAction G A
    f : Prod G G → A
    hf : groupCohomology.IsTwoCocycle f
    g : G
    this : Eq (HAdd.hAdd (f { fst := 1, snd := 1 }) (f { fst := g, snd := Inv.inv  …
    ⊢ Eq (HSub.hSub (HSMul.hSMul g (f { fst := Inv.inv g, snd := g })) (f { fst := …
  -/
  exact sub_eq_sub_iff_add_eq_add.2 this.symm
  /-
    🎉 no goals
  -/


/-- A function `f : G → A` satisfies the 1-coboundary condition if there's `x : A` such that
`g • x - x = f(g)` for all `g : G`. -/
def IsOneCoboundary (f : G → A) : Prop := ∃ x : A, ∀ g : G, g • x - x = f g


/-- A function `f : G × G → A` satisfies the 2-coboundary condition if there's `x : G → A` such
that `g • x(h) - x(gh) + x(g) = f(g, h)` for all `g, h : G`. -/
def IsTwoCoboundary (f : G × G → A) : Prop :=
  ∃ x : G → A, ∀ g h : G, g • x h - x (g * h) + x g = f (g, h)


/-- Given a `k`-module `A` with a compatible `DistribMulAction` of `G`, and a function
`f : G → A` satisfying the 1-cocycle condition, produces a 1-cocycle for the representation on
`A` induced by the `DistribMulAction`. -/
def oneCocyclesOfIsOneCocycle {f : G → A} (hf : IsOneCocycle f) :
    oneCocycles (Rep.ofDistribMulAction k G A) :=
  ⟨f, (mem_oneCocycles_iff (A := Rep.ofDistribMulAction k G A) f).2 hf⟩


theorem isOneCocycle_of_oneCocycles (f : oneCocycles (Rep.ofDistribMulAction k G A)) :
    IsOneCocycle (A := A) f := (mem_oneCocycles_iff f).1 f.2


/-- Given a `k`-module `A` with a compatible `DistribMulAction` of `G`, and a function
`f : G → A` satisfying the 1-coboundary condition, produces a 1-coboundary for the representation
on `A` induced by the `DistribMulAction`. -/
def oneCoboundariesOfIsOneCoboundary {f : G → A} (hf : IsOneCoboundary f) :
    oneCoboundaries (Rep.ofDistribMulAction k G A) :=
  oneCoboundariesOfMemRange ⟨hf.choose, funext hf.choose_spec⟩


theorem isOneCoboundary_of_oneCoboundaries (f : oneCoboundaries (Rep.ofDistribMulAction k G A)) :
    IsOneCoboundary (A := A) f.1.1 := by
  /-
    k G A : Type u
    inst✝⁵ : CommRing k
    inst✝⁴ : Group G
    inst✝³ : AddCommGroup A
    inst✝² : Module k A
    inst✝¹ : DistribMulAction G A
    inst✝ : SMulCommClass G k A
    f : Subtype fun x => Membership.mem (groupCohomology.oneCoboundaries (Rep.ofDi …
    ⊢ groupCohomology.IsOneCoboundary ↑↑f
  -/
  rcases mem_range_of_mem_oneCoboundaries f.2 with ⟨x, hx⟩
  /-
    case intro
    k G A : Type u
    inst✝⁵ : CommRing k
    inst✝⁴ : Group G
    inst✝³ : AddCommGroup A
    inst✝² : Module k A
    inst✝¹ : DistribMulAction G A
    inst✝ : SMulCommClass G k A
    f : Subtype fun x => Membership.mem (groupCohomology.oneCoboundaries (Rep.ofDi …
    x : CoeSort.coe (Rep.ofDistribMulAction k G A)
    hx : Eq ((groupCohomology.dZero (Rep.ofDistribMulAction k G A)) x) ↑↑f
    ⊢ groupCohomology.IsOneCoboundary ↑↑f
  -/
  exact ⟨x, by rw [← hx]; intro g; rfl⟩
  /-
    🎉 no goals
  -/


/-- Given a `k`-module `A` with a compatible `DistribMulAction` of `G`, and a function
`f : G × G → A` satisfying the 2-cocycle condition, produces a 2-cocycle for the representation on
`A` induced by the `DistribMulAction`. -/
def twoCocyclesOfIsTwoCocycle {f : G × G → A} (hf : IsTwoCocycle f) :
    twoCocycles (Rep.ofDistribMulAction k G A) :=
  ⟨f, (mem_twoCocycles_iff (A := Rep.ofDistribMulAction k G A) f).2 hf⟩


theorem isTwoCocycle_of_twoCocycles (f : twoCocycles (Rep.ofDistribMulAction k G A)) :
    IsTwoCocycle (A := A) f := (mem_twoCocycles_iff f).1 f.2


/-- Given a `k`-module `A` with a compatible `DistribMulAction` of `G`, and a function
`f : G × G → A` satisfying the 2-coboundary condition, produces a 2-coboundary for the
representation on `A` induced by the `DistribMulAction`. -/
def twoCoboundariesOfIsTwoCoboundary {f : G × G → A} (hf : IsTwoCoboundary f) :
    twoCoboundaries (Rep.ofDistribMulAction k G A) :=
  twoCoboundariesOfMemRange (⟨hf.choose,funext fun g ↦ hf.choose_spec g.1 g.2⟩)


theorem isTwoCoboundary_of_twoCoboundaries (f : twoCoboundaries (Rep.ofDistribMulAction k G A)) :
    IsTwoCoboundary (A := A) f.1.1 := by
  /-
    k G A : Type u
    inst✝⁵ : CommRing k
    inst✝⁴ : Group G
    inst✝³ : AddCommGroup A
    inst✝² : Module k A
    inst✝¹ : DistribMulAction G A
    inst✝ : SMulCommClass G k A
    f : Subtype fun x => Membership.mem (groupCohomology.twoCoboundaries (Rep.ofDi …
    ⊢ groupCohomology.IsTwoCoboundary ↑↑f
  -/
  rcases mem_range_of_mem_twoCoboundaries f.2 with ⟨x, hx⟩
  /-
    case intro
    k G A : Type u
    inst✝⁵ : CommRing k
    inst✝⁴ : Group G
    inst✝³ : AddCommGroup A
    inst✝² : Module k A
    inst✝¹ : DistribMulAction G A
    inst✝ : SMulCommClass G k A
    f : Subtype fun x => Membership.mem (groupCohomology.twoCoboundaries (Rep.ofDi …
    x : G → CoeSort.coe (Rep.ofDistribMulAction k G A)
    hx : Eq ((groupCohomology.dOne (Rep.ofDistribMulAction k G A)) x) ((groupCohom …
    ⊢ groupCohomology.IsTwoCoboundary ↑↑f
  -/
  exact ⟨x, fun g h => funext_iff.1 hx (g, h)⟩
  /-
    🎉 no goals
  -/


/-- A function `f : G → M` satisfies the multiplicative 1-cocycle condition if
`f(gh) = g • f(h) * f(g)` for all `g, h : G`. -/
def IsMulOneCocycle (f : G → M) : Prop := ∀ g h : G, f (g * h) = g • f h * f g


/-- A function `f : G × G → M` satisfies the multiplicative 2-cocycle condition if
`f(gh, j) * f(g, h) = g • f(h, j) * f(g, hj)` for all `g, h : G`. -/
def IsMulTwoCocycle (f : G × G → M) : Prop :=
  ∀ g h j : G, f (g * h, j) * f (g, h) = g • (f (h, j)) * f (g, h * j)


theorem map_one_of_isMulOneCocycle {f : G → M} (hf : IsMulOneCocycle f) :
    f 1 = 1 := by
  /-
    G : Type u_1
    M : Type u_2
    inst✝² : Monoid G
    inst✝¹ : CommGroup M
    inst✝ : MulAction G M
    f : G → M
    hf : groupCohomology.IsMulOneCocycle f
    ⊢ Eq (f 1) 1
  -/
  simpa only [mul_one, one_smul, self_eq_mul_right] using hf 1 1
  /-
    🎉 no goals
  -/


theorem map_one_fst_of_isMulTwoCocycle {f : G × G → M} (hf : IsMulTwoCocycle f) (g : G) :
    f (1, g) = f (1, 1) := by
  /-
    G : Type u_1
    M : Type u_2
    inst✝² : Monoid G
    inst✝¹ : CommGroup M
    inst✝ : MulAction G M
    f : Prod G G → M
    hf : groupCohomology.IsMulTwoCocycle f
    g : G
    ⊢ Eq (f { fst := 1, snd := g }) (f { fst := 1, snd := 1 })
  -/
  simpa only [one_smul, one_mul, mul_one, mul_right_inj] using (hf 1 1 g).symm
  /-
    🎉 no goals
  -/


theorem map_one_snd_of_isMulTwoCocycle {f : G × G → M} (hf : IsMulTwoCocycle f) (g : G) :
    f (g, 1) = g • f (1, 1) := by
  /-
    G : Type u_1
    M : Type u_2
    inst✝² : Monoid G
    inst✝¹ : CommGroup M
    inst✝ : MulAction G M
    f : Prod G G → M
    hf : groupCohomology.IsMulTwoCocycle f
    g : G
    ⊢ Eq (f { fst := g, snd := 1 }) (HSMul.hSMul g (f { fst := 1, snd := 1 }))
  -/
  simpa only [mul_one, mul_left_inj] using hf g 1 1
  /-
    🎉 no goals
  -/


@[scoped simp] theorem map_inv_of_isMulOneCocycle {f : G → M} (hf : IsMulOneCocycle f) (g : G) :
    g • f g⁻¹ = (f g)⁻¹ := by
  /-
    G : Type u_1
    M : Type u_2
    inst✝² : Group G
    inst✝¹ : CommGroup M
    inst✝ : MulAction G M
    f : G → M
    hf : groupCohomology.IsMulOneCocycle f
    g : G
    ⊢ Eq (HSMul.hSMul g (f (Inv.inv g))) (Inv.inv (f g))
  -/
  rw [← mul_eq_one_iff_eq_inv, ← map_one_of_isMulOneCocycle hf, ← mul_inv_cancel g, hf g g⁻¹]
  /-
    🎉 no goals
  -/


theorem smul_map_inv_div_map_inv_of_isMulTwoCocycle
    {f : G × G → M} (hf : IsMulTwoCocycle f) (g : G) :
    g • f (g⁻¹, g) / f (g, g⁻¹) = f (1, 1) / f (g, 1) := by
  /-
    G : Type u_1
    M : Type u_2
    inst✝² : Group G
    inst✝¹ : CommGroup M
    inst✝ : MulAction G M
    f : Prod G G → M
    hf : groupCohomology.IsMulTwoCocycle f
    g : G
    ⊢ Eq (HDiv.hDiv (HSMul.hSMul g (f { fst := Inv.inv g, snd := g })) (f { fst := …
  -/
  have := hf g g⁻¹ g
  /-
    G : Type u_1
    M : Type u_2
    inst✝² : Group G
    inst✝¹ : CommGroup M
    inst✝ : MulAction G M
    f : Prod G G → M
    hf : groupCohomology.IsMulTwoCocycle f
    g : G
    this : Eq (HMul.hMul (f { fst := HMul.hMul g (Inv.inv g), snd := g }) (f { fst …
    ⊢ Eq (HDiv.hDiv (HSMul.hSMul g (f { fst := Inv.inv g, snd := g })) (f { fst := …
  -/
  simp only [mul_inv_cancel, inv_mul_cancel, map_one_fst_of_isMulTwoCocycle hf g] at this
  /-
    G : Type u_1
    M : Type u_2
    inst✝² : Group G
    inst✝¹ : CommGroup M
    inst✝ : MulAction G M
    f : Prod G G → M
    hf : groupCohomology.IsMulTwoCocycle f
    g : G
    this : Eq (HMul.hMul (f { fst := 1, snd := 1 }) (f { fst := g, snd := Inv.inv  …
    ⊢ Eq (HDiv.hDiv (HSMul.hSMul g (f { fst := Inv.inv g, snd := g })) (f { fst := …
  -/
  exact div_eq_div_iff_mul_eq_mul.2 this.symm
  /-
    🎉 no goals
  -/


/-- A function `f : G → M` satisfies the multiplicative 1-coboundary condition if there's `x : M`
such that `g • x / x = f(g)` for all `g : G`. -/
def IsMulOneCoboundary (f : G → M) : Prop := ∃ x : M, ∀ g : G, g • x / x = f g


/-- A function `f : G × G → M` satisfies the 2-coboundary condition if there's `x : G → M` such
that `g • x(h) / x(gh) * x(g) = f(g, h)` for all `g, h : G`. -/
def IsMulTwoCoboundary (f : G × G → M) : Prop :=
  ∃ x : G → M, ∀ g h : G, g • x h / x (g * h) * x g = f (g, h)


/-- Given an abelian group `M` with a `MulDistribMulAction` of `G`, and a function
`f : G → M` satisfying the multiplicative 1-cocycle condition, produces a 1-cocycle for the
representation on `Additive M` induced by the `MulDistribMulAction`. -/
def oneCocyclesOfIsMulOneCocycle {f : G → M} (hf : IsMulOneCocycle f) :
    oneCocycles (Rep.ofMulDistribMulAction G M) :=
  ⟨Additive.ofMul ∘ f, (mem_oneCocycles_iff (A := Rep.ofMulDistribMulAction G M) f).2 hf⟩


theorem isMulOneCocycle_of_oneCocycles (f : oneCocycles (Rep.ofMulDistribMulAction G M)) :
    IsMulOneCocycle (M := M) (Additive.toMul ∘ f) := (mem_oneCocycles_iff f).1 f.2


/-- Given an abelian group `M` with a `MulDistribMulAction` of `G`, and a function
`f : G → M` satisfying the multiplicative 1-coboundary condition, produces a
1-coboundary for the representation on `Additive M` induced by the `MulDistribMulAction`. -/
def oneCoboundariesOfIsMulOneCoboundary {f : G → M} (hf : IsMulOneCoboundary f) :
    oneCoboundaries (Rep.ofMulDistribMulAction G M) :=
  oneCoboundariesOfMemRange (f := Additive.ofMul ∘ f) ⟨hf.choose, funext hf.choose_spec⟩


theorem isMulOneCoboundary_of_oneCoboundaries
    (f : oneCoboundaries (Rep.ofMulDistribMulAction G M)) :
    IsMulOneCoboundary (M := M) (Additive.ofMul ∘ f.1.1) := by
  /-
    G M : Type
    inst✝² : Group G
    inst✝¹ : CommGroup M
    inst✝ : MulDistribMulAction G M
    f : Subtype fun x => Membership.mem (groupCohomology.oneCoboundaries (Rep.ofMu …
    ⊢ groupCohomology.IsMulOneCoboundary (Function.comp ⇑Additive.ofMul ↑↑f)
  -/
  rcases mem_range_of_mem_oneCoboundaries f.2 with ⟨x, hx⟩
  /-
    case intro
    G M : Type
    inst✝² : Group G
    inst✝¹ : CommGroup M
    inst✝ : MulDistribMulAction G M
    f : Subtype fun x => Membership.mem (groupCohomology.oneCoboundaries (Rep.ofMu …
    x : CoeSort.coe (Rep.ofMulDistribMulAction G M)
    hx : Eq ((groupCohomology.dZero (Rep.ofMulDistribMulAction G M)) x) ↑↑f
    ⊢ groupCohomology.IsMulOneCoboundary (Function.comp ⇑Additive.ofMul ↑↑f)
  -/
  exact ⟨x, by rw [← hx]; intro g; rfl⟩
  /-
    🎉 no goals
  -/


/-- Given an abelian group `M` with a `MulDistribMulAction` of `G`, and a function
`f : G × G → M` satisfying the multiplicative 2-cocycle condition, produces a 2-cocycle for the
representation on `Additive M` induced by the `MulDistribMulAction`. -/
def twoCocyclesOfIsMulTwoCocycle {f : G × G → M} (hf : IsMulTwoCocycle f) :
    twoCocycles (Rep.ofMulDistribMulAction G M) :=
  ⟨Additive.ofMul ∘ f, (mem_twoCocycles_iff (A := Rep.ofMulDistribMulAction G M) f).2 hf⟩


theorem isMulTwoCocycle_of_twoCocycles (f : twoCocycles (Rep.ofMulDistribMulAction G M)) :
    IsMulTwoCocycle (M := M) (Additive.toMul ∘ f) := (mem_twoCocycles_iff f).1 f.2


/-- Given an abelian group `M` with a `MulDistribMulAction` of `G`, and a function
`f : G × G → M` satisfying the multiplicative 2-coboundary condition, produces a
2-coboundary for the representation on `M` induced by the `MulDistribMulAction`. -/
def twoCoboundariesOfIsMulTwoCoboundary {f : G × G → M} (hf : IsMulTwoCoboundary f) :
    twoCoboundaries (Rep.ofMulDistribMulAction G M) :=
  twoCoboundariesOfMemRange ⟨hf.choose, funext fun g ↦ hf.choose_spec g.1 g.2⟩


theorem isMulTwoCoboundary_of_twoCoboundaries
    (f : twoCoboundaries (Rep.ofMulDistribMulAction G M)) :
    IsMulTwoCoboundary (M := M) (Additive.toMul ∘ f.1.1) := by
  /-
    G M : Type
    inst✝² : Group G
    inst✝¹ : CommGroup M
    inst✝ : MulDistribMulAction G M
    f : Subtype fun x => Membership.mem (groupCohomology.twoCoboundaries (Rep.ofMu …
    ⊢ groupCohomology.IsMulTwoCoboundary (Function.comp ⇑Additive.toMul ↑↑f)
  -/
  rcases mem_range_of_mem_twoCoboundaries f.2 with ⟨x, hx⟩
  /-
    case intro
    G M : Type
    inst✝² : Group G
    inst✝¹ : CommGroup M
    inst✝ : MulDistribMulAction G M
    f : Subtype fun x => Membership.mem (groupCohomology.twoCoboundaries (Rep.ofMu …
    x : G → CoeSort.coe (Rep.ofMulDistribMulAction G M)
    hx : Eq ((groupCohomology.dOne (Rep.ofMulDistribMulAction G M)) x) ((groupCoho …
    ⊢ groupCohomology.IsMulTwoCoboundary (Function.comp ⇑Additive.toMul ↑↑f)
  -/
  exact ⟨x, fun g h => funext_iff.1 hx (g, h)⟩
  /-
    🎉 no goals
  -/


/-- We define the 0th group cohomology of a `k`-linear `G`-representation `A`, `H⁰(G, A)`, to be
the invariants of the representation, `Aᴳ`. -/
abbrev H0 := A.ρ.invariants


/-- We define the 1st group cohomology of a `k`-linear `G`-representation `A`, `H¹(G, A)`, to be
1-cocycles (i.e. `Z¹(G, A) := Ker(d¹ : Fun(G, A) → Fun(G², A)`) modulo 1-coboundaries
(i.e. `B¹(G, A) := Im(d⁰: A → Fun(G, A))`). -/
abbrev H1 := oneCocycles A ⧸ oneCoboundaries A


/-- The quotient map `Z¹(G, A) → H¹(G, A).` -/
def H1_π : oneCocycles A →ₗ[k] H1 A := (oneCoboundaries A).mkQ


/-- We define the 2nd group cohomology of a `k`-linear `G`-representation `A`, `H²(G, A)`, to be
2-cocycles (i.e. `Z²(G, A) := Ker(d² : Fun(G², A) → Fun(G³, A)`) modulo 2-coboundaries
(i.e. `B²(G, A) := Im(d¹: Fun(G, A) → Fun(G², A))`). -/
abbrev H2 := twoCocycles A ⧸ twoCoboundaries A


/-- The quotient map `Z²(G, A) → H²(G, A).` -/
def H2_π : twoCocycles A →ₗ[k] H2 A := (twoCoboundaries A).mkQ


/-- When the representation on `A` is trivial, then `H⁰(G, A)` is all of `A.` -/
def H0LequivOfIsTrivial [A.IsTrivial] :
    H0 A ≃ₗ[k] A := LinearEquiv.ofTop _ (invariants_eq_top A.ρ)


@[simp] theorem H0LequivOfIsTrivial_eq_subtype [A.IsTrivial] :
    H0LequivOfIsTrivial A = A.ρ.invariants.subtype := rfl


theorem H0LequivOfIsTrivial_apply [A.IsTrivial] (x : H0 A) :
    H0LequivOfIsTrivial A x = x := rfl


@[simp] theorem H0LequivOfIsTrivial_symm_apply [A.IsTrivial] (x : A) :
    (H0LequivOfIsTrivial A).symm x = x := rfl


/-- When `A : Rep k G` is a trivial representation of `G`, `H¹(G, A)` is isomorphic to the
group homs `G → A`. -/
def H1LequivOfIsTrivial [A.IsTrivial] :
    H1 A ≃ₗ[k] Additive G →+ A :=
  (Submodule.quotEquivOfEqBot _ (oneCoboundaries_eq_bot_of_isTrivial A)).trans
    (oneCocyclesLequivOfIsTrivial A)


theorem H1LequivOfIsTrivial_comp_H1_π [A.IsTrivial] :
    (H1LequivOfIsTrivial A).comp (H1_π A) = oneCocyclesLequivOfIsTrivial A := by
  /-
    k G : Type u
    inst✝² : CommRing k
    inst✝¹ : Group G
    A : Rep k G
    inst✝ : A.IsTrivial
    ⊢ Eq ((↑(groupCohomology.H1LequivOfIsTrivial A)).comp (groupCohomology.H1_π A) …
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


@[simp] theorem H1LequivOfIsTrivial_H1_π_apply_apply
    [A.IsTrivial] (f : oneCocycles A) (x : Additive G) :
    H1LequivOfIsTrivial A (H1_π A f) x = f x.toMul := rfl


@[simp] theorem H1LequivOfIsTrivial_symm_apply [A.IsTrivial] (f : Additive G →+ A) :
    (H1LequivOfIsTrivial A).symm f = H1_π A ((oneCocyclesLequivOfIsTrivial A).symm f) :=
  rfl


lemma dZero_comp_H0_subtype : dZero A ∘ₗ (H0 A).subtype = 0 := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq ((groupCohomology.dZero A).comp (groupCohomology.H0 A).subtype) 0
  -/
  ext ⟨x, hx⟩ g
  /-
    case h.mk.h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : CoeSort.coe A
    hx : Membership.mem (groupCohomology.H0 A) x
    g : G
    ⊢ Eq (((groupCohomology.dZero A).comp (groupCohomology.H0 A).subtype) ⟨x, hx⟩  …
  -/
  replace hx := hx g
  /-
    case h.mk.h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : CoeSort.coe A
    hx✝ : Membership.mem (groupCohomology.H0 A) x
    g : G
    hx : Eq ((A.ρ g) x) x
    ⊢ Eq (((groupCohomology.dZero A).comp (groupCohomology.H0 A).subtype) ⟨x, hx✝⟩ …
  -/
  rw [← sub_eq_zero] at hx
  /-
    case h.mk.h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : CoeSort.coe A
    hx✝ : Membership.mem (groupCohomology.H0 A) x
    g : G
    hx : Eq (HSub.hSub ((A.ρ g) x) x) 0
    ⊢ Eq (((groupCohomology.dZero A).comp (groupCohomology.H0 A).subtype) ⟨x, hx✝⟩ …
  -/
  exact hx
  /-
    🎉 no goals
  -/


/-- The (exact) short complex `A.ρ.invariants ⟶ A ⟶ (G → A)`. -/
def shortComplexH0 : ShortComplex (ModuleCat k) :=
  ShortComplex.moduleCatMk _ _ (dZero_comp_H0_subtype A)


instance : Mono (shortComplexH0 A).f := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ CategoryTheory.Mono (groupCohomology.shortComplexH0 A).f
  -/
  rw [ModuleCat.mono_iff_injective]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Function.Injective ⇑(groupCohomology.shortComplexH0 A).f.hom
  -/
  apply Submodule.injective_subtype
  /-
    🎉 no goals
  -/


lemma shortComplexH0_exact : (shortComplexH0 A).Exact := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ (groupCohomology.shortComplexH0 A).Exact
  -/
  rw [ShortComplex.moduleCat_exact_iff]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ ∀ (x₂ : ↑(groupCohomology.shortComplexH0 A).X₂), Eq ((groupCohomology.shortC …
  -/
  intro (x : A) (hx : dZero _ x = 0)
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : CoeSort.coe A
    hx : Eq ((groupCohomology.dZero A) x) 0
    ⊢ Exists fun x₁ => Eq ((groupCohomology.shortComplexH0 A).f.hom x₁) x
  -/
  refine ⟨⟨x, fun g => ?_⟩, rfl⟩
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : CoeSort.coe A
    hx : Eq ((groupCohomology.dZero A) x) 0
    g : G
    ⊢ Eq ((A.ρ g) x) x
  -/
  rw [← sub_eq_zero]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    x : CoeSort.coe A
    hx : Eq ((groupCohomology.dZero A) x) 0
    g : G
    ⊢ Eq (HSub.hSub ((A.ρ g) x) x) 0
  -/
  exact congr_fun hx g
  /-
    🎉 no goals
  -/


/-- The arrow `A --dZero--> Fun(G, A)` is isomorphic to the differential
`(inhomogeneousCochains A).d 0 1` of the complex of inhomogeneous cochains of `A`. -/
@[simps! hom_left hom_right inv_left inv_right]
def dZeroArrowIso : Arrow.mk ((inhomogeneousCochains A).d 0 1) ≅
    Arrow.mk (ModuleCat.ofHom (dZero A)) :=
  Arrow.isoMk (zeroCochainsLequiv A).toModuleIso
    (oneCochainsLequiv A).toModuleIso (ModuleCat.hom_ext (dZero_comp_eq A))


/-- The 0-cocycles of the complex of inhomogeneous cochains of `A` are isomorphic to
`A.ρ.invariants`, which is a simpler type. -/
def isoZeroCocycles : cocycles A 0 ≅ ModuleCat.of k A.ρ.invariants :=
  KernelFork.mapIsoOfIsLimit
                                                      /-
                                                        k G : Type u
                                                        inst✝¹ : CommRing k
                                                        inst✝ : Group G
                                                        A : Rep k G
                                                        ⊢ Eq ((ComplexShape.up Nat).next 0) 1
                                                      -/
    ((inhomogeneousCochains A).cyclesIsKernel 0 1 (by simp)) (shortComplexH0_exact A).fIsKernel
                                                      /-
                                                        🎉 no goals
                                                      -/
      (dZeroArrowIso A)


lemma isoZeroCocycles_hom_comp_subtype :
    (isoZeroCocycles A).hom ≫ ModuleCat.ofHom A.ρ.invariants.subtype =
      iCocycles A 0 ≫ (zeroCochainsLequiv A).toModuleIso.hom := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (groupCohomology.isoZeroCocycles A).h …
  -/
  dsimp [isoZeroCocycles]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.KernelFork.of …
  -/
  apply KernelFork.mapOfIsLimit_ι
  /-
    🎉 no goals
  -/


/-- The 0th group cohomology of `A`, defined as the 0th cohomology of the complex of inhomogeneous
cochains, is isomorphic to the invariants of the representation on `A`. -/
def isoH0 : groupCohomology A 0 ≅ ModuleCat.of k (H0 A) :=
  (CochainComplex.isoHomologyπ₀ _).symm ≪≫ isoZeroCocycles A


lemma groupCohomologyπ_comp_isoH0_hom  :
    groupCohomologyπ A 0 ≫ (isoH0 A).hom = (isoZeroCocycles A).hom := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (groupCohomologyπ A 0) (groupCohomolo …
  -/
  simp [isoH0]
  /-
    🎉 no goals
  -/


/-- The short complex `A --dZero--> Fun(G, A) --dOne--> Fun(G × G, A)`. -/
def shortComplexH1 : ShortComplex (ModuleCat k) :=
  moduleCatMk (dZero A) (dOne A) (dOne_comp_dZero A)


/-- The short complex `A --dZero--> Fun(G, A) --dOne--> Fun(G × G, A)` is isomorphic to the 1st
short complex associated to the complex of inhomogeneous cochains of `A`. -/
@[simps! hom inv]
def shortComplexH1Iso : (inhomogeneousCochains A).sc' 0 1 2 ≅ shortComplexH1 A :=
    isoMk (zeroCochainsLequiv A).toModuleIso (oneCochainsLequiv A).toModuleIso
      (twoCochainsLequiv A).toModuleIso
        (ModuleCat.hom_ext (dZero_comp_eq A))
        (ModuleCat.hom_ext (dOne_comp_eq A))


/-- The 1-cocycles of the complex of inhomogeneous cochains of `A` are isomorphic to
`oneCocycles A`, which is a simpler type. -/
def isoOneCocycles : cocycles A 1 ≅ ModuleCat.of k (oneCocycles A) :=
                                                   /-
                                                     k G : Type u
                                                     inst✝¹ : CommRing k
                                                     inst✝ : Group G
                                                     A : Rep k G
                                                     ⊢ Eq ((ComplexShape.up Nat).prev 1) 0
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  (inhomogeneousCochains A).cyclesIsoSc' _ _ _ (by aesop) (by aesop) ≪≫
                                                              /-
                                                                🎉 no goals
                                                              -/
    cyclesMapIso (shortComplexH1Iso A) ≪≫ (shortComplexH1 A).moduleCatCyclesIso


lemma isoOneCocycles_hom_comp_subtype :
    (isoOneCocycles A).hom ≫ ModuleCat.ofHom (oneCocycles A).subtype =
      iCocycles A 1 ≫ (oneCochainsLequiv A).toModuleIso.hom := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (groupCohomology.isoOneCocycles A).ho …
  -/
  dsimp [isoOneCocycles]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [Category.assoc, Category.assoc]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cyclesIsoSc' (gro …
  -/
  erw [(shortComplexH1 A).moduleCatCyclesIso_hom_subtype]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cyclesIsoSc' (gro …
  -/
  rw [cyclesMap_i, HomologicalComplex.cyclesIsoSc'_hom_iCycles_assoc]
  /-
    🎉 no goals
  -/


lemma toCocycles_comp_isoOneCocycles_hom :
    toCocycles A 0 1 ≫ (isoOneCocycles A).hom =
      (zeroCochainsLequiv A).toModuleIso.hom ≫
        ModuleCat.ofHom (shortComplexH1 A).moduleCatToCycles := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (groupCohomology.toCocycles A 0 1) (g …
  -/
  simp [isoOneCocycles]
  /-
    🎉 no goals
  -/


/-- The 1st group cohomology of `A`, defined as the 1st cohomology of the complex of inhomogeneous
cochains, is isomorphic to `oneCocycles A ⧸ oneCoboundaries A`, which is a simpler type. -/
def isoH1 : groupCohomology A 1 ≅ ModuleCat.of k (H1 A) :=
                                                     /-
                                                       k G : Type u
                                                       inst✝¹ : CommRing k
                                                       inst✝ : Group G
                                                       A : Rep k G
                                                       ⊢ Eq ((ComplexShape.up Nat).prev 1) 0
                                                     -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  (inhomogeneousCochains A).homologyIsoSc' _ _ _ (by aesop) (by aesop) ≪≫
                                                                /-
                                                                  🎉 no goals
                                                                -/
    homologyMapIso (shortComplexH1Iso A) ≪≫ (shortComplexH1 A).moduleCatHomologyIso


lemma groupCohomologyπ_comp_isoH1_hom  :
    groupCohomologyπ A 1 ≫ (isoH1 A).hom =
      (isoOneCocycles A).hom ≫ (shortComplexH1 A).moduleCatHomologyπ := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (groupCohomologyπ A 1) (groupCohomolo …
  -/
  simp [isoH1, isoOneCocycles]
  /-
    🎉 no goals
  -/


/-- The short complex `Fun(G, A) --dOne--> Fun(G × G, A) --dTwo--> Fun(G × G × G, A)`. -/
def shortComplexH2 : ShortComplex (ModuleCat k) :=
  moduleCatMk (dOne A) (dTwo A) (dTwo_comp_dOne A)


/-- The short complex `Fun(G, A) --dOne--> Fun(G × G, A) --dTwo--> Fun(G × G × G, A)` is
isomorphic to the 2nd short complex associated to the complex of inhomogeneous cochains of `A`. -/
@[simps! hom inv]
def shortComplexH2Iso :
    (inhomogeneousCochains A).sc' 1 2 3 ≅ shortComplexH2 A :=
  isoMk (oneCochainsLequiv A).toModuleIso (twoCochainsLequiv A).toModuleIso
    (threeCochainsLequiv A).toModuleIso
      (ModuleCat.hom_ext (dOne_comp_eq A))
      (ModuleCat.hom_ext (dTwo_comp_eq A))


/-- The 2-cocycles of the complex of inhomogeneous cochains of `A` are isomorphic to
`twoCocycles A`, which is a simpler type. -/
def isoTwoCocycles : cocycles A 2 ≅ ModuleCat.of k (twoCocycles A) :=
                                                   /-
                                                     k G : Type u
                                                     inst✝¹ : CommRing k
                                                     inst✝ : Group G
                                                     A : Rep k G
                                                     ⊢ Eq ((ComplexShape.up Nat).prev 2) 1
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  (inhomogeneousCochains A).cyclesIsoSc' _ _ _ (by aesop) (by aesop) ≪≫
                                                              /-
                                                                🎉 no goals
                                                              -/
    cyclesMapIso (shortComplexH2Iso A) ≪≫ (shortComplexH2 A).moduleCatCyclesIso


lemma isoTwoCocycles_hom_comp_subtype :
    (isoTwoCocycles A).hom ≫ ModuleCat.ofHom (twoCocycles A).subtype =
      iCocycles A 2 ≫ (twoCochainsLequiv A).toModuleIso.hom := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (groupCohomology.isoTwoCocycles A).ho …
  -/
  dsimp [isoTwoCocycles]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [Category.assoc, Category.assoc]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cyclesIsoSc' (gro …
  -/
  erw [(shortComplexH2 A).moduleCatCyclesIso_hom_subtype]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cyclesIsoSc' (gro …
  -/
  rw [cyclesMap_i, HomologicalComplex.cyclesIsoSc'_hom_iCycles_assoc]
  /-
    🎉 no goals
  -/


lemma toCocycles_comp_isoTwoCocycles_hom :
    toCocycles A 1 2 ≫ (isoTwoCocycles A).hom =
      (oneCochainsLequiv A).toModuleIso.hom ≫
        ModuleCat.ofHom (shortComplexH2 A).moduleCatToCycles := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (groupCohomology.toCocycles A 1 2) (g …
  -/
  simp [isoTwoCocycles]
  /-
    🎉 no goals
  -/


/-- The 2nd group cohomology of `A`, defined as the 2nd cohomology of the complex of inhomogeneous
cochains, is isomorphic to `twoCocycles A ⧸ twoCoboundaries A`, which is a simpler type. -/
def isoH2 : groupCohomology A 2 ≅ ModuleCat.of k (H2 A) :=
                                                     /-
                                                       k G : Type u
                                                       inst✝¹ : CommRing k
                                                       inst✝ : Group G
                                                       A : Rep k G
                                                       ⊢ Eq ((ComplexShape.up Nat).prev 2) 1
                                                     -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  (inhomogeneousCochains A).homologyIsoSc' _ _ _ (by aesop) (by aesop) ≪≫
                                                                /-
                                                                  🎉 no goals
                                                                -/
    homologyMapIso (shortComplexH2Iso A) ≪≫ (shortComplexH2 A).moduleCatHomologyIso


lemma groupCohomologyπ_comp_isoH2_hom  :
    groupCohomologyπ A 2 ≫ (isoH2 A).hom =
      (isoTwoCocycles A).hom ≫ (shortComplexH2 A).moduleCatHomologyπ := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (groupCohomologyπ A 2) (groupCohomolo …
  -/
  simp [isoH2, isoTwoCocycles]
  /-
    🎉 no goals
  -/


