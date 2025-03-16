/-- A representation of `G` on the `k`-module `V` is a homomorphism `G →* (V →ₗ[k] V)`.
-/
abbrev Representation :=
  G →* V →ₗ[k] V


/-- The trivial representation of `G` on a `k`-module V.
-/
def trivial : Representation k G V :=
  1

-- Porting note: why is `V` implicit

theorem trivial_def (g : G) (v : V) : trivial k (V := V) g v = v :=
  rfl


/-- A predicate for representations that fix every element. -/
class IsTrivial (ρ : Representation k G V) : Prop where
  out : ∀ g x, ρ g x = x := by aesop


instance : IsTrivial (trivial k (G := G) (V := V)) where


@[simp] theorem apply_eq_self
    (ρ : Representation k G V) (g : G) (x : V) [h : IsTrivial ρ] :
    ρ g x = x := h.out g x


/-- A `k`-linear representation of `G` on `V` can be thought of as
an algebra map from `MonoidAlgebra k G` into the `k`-linear endomorphisms of `V`.
-/
noncomputable def asAlgebraHom : MonoidAlgebra k G →ₐ[k] Module.End k V :=
  (lift k G _) ρ


theorem asAlgebraHom_def : asAlgebraHom ρ = (lift k G _) ρ :=
  rfl


@[simp]
theorem asAlgebraHom_single (g : G) (r : k) : asAlgebraHom ρ (Finsupp.single g r) = r • ρ g := by
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    inst✝³ : CommSemiring k
    inst✝² : Monoid G
    inst✝¹ : AddCommMonoid V
    inst✝ : Module k V
    ρ : Representation k G V
    g : G
    r : k
    ⊢ Eq (ρ.asAlgebraHom (Finsupp.single g r)) (HSMul.hSMul r (ρ g))
  -/
  simp only [asAlgebraHom_def, MonoidAlgebra.lift_single]
  /-
    🎉 no goals
  -/


                                                                                          /-
                                                                                            k : Type u_1
                                                                                            G : Type u_2
                                                                                            V : Type u_3
                                                                                            inst✝³ : CommSemiring k
                                                                                            inst✝² : Monoid G
                                                                                            inst✝¹ : AddCommMonoid V
                                                                                            inst✝ : Module k V
                                                                                            ρ : Representation k G V
                                                                                            g : G
                                                                                            ⊢ Eq (ρ.asAlgebraHom (Finsupp.single g 1)) (ρ g)
                                                                                          -/
theorem asAlgebraHom_single_one (g : G) : asAlgebraHom ρ (Finsupp.single g 1) = ρ g := by simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


theorem asAlgebraHom_of (g : G) : asAlgebraHom ρ (of k G g) = ρ g := by
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    inst✝³ : CommSemiring k
    inst✝² : Monoid G
    inst✝¹ : AddCommMonoid V
    inst✝ : Module k V
    ρ : Representation k G V
    g : G
    ⊢ Eq (ρ.asAlgebraHom ((MonoidAlgebra.of k G) g)) (ρ g)
  -/
  simp only [MonoidAlgebra.of_apply, asAlgebraHom_single, one_smul]
  /-
    🎉 no goals
  -/


/-- If `ρ : Representation k G V`, then `ρ.asModule` is a type synonym for `V`,
which we equip with an instance `Module (MonoidAlgebra k G) ρ.asModule`.

You should use `asModuleEquiv : ρ.asModule ≃+ V` to translate terms.
-/
@[nolint unusedArguments]
def asModule (_ : Representation k G V) :=
  V

-- Porting note: no derive handler

instance : AddCommMonoid (ρ.asModule) := inferInstanceAs <| AddCommMonoid V


instance : Inhabited ρ.asModule where
  default := 0


/-- A `k`-linear representation of `G` on `V` can be thought of as
a module over `MonoidAlgebra k G`.
-/
noncomputable instance asModuleModule : Module (MonoidAlgebra k G) ρ.asModule :=
  Module.compHom V (asAlgebraHom ρ).toRingHom

-- Porting note: ρ.asModule doesn't unfold now

instance : Module k ρ.asModule := inferInstanceAs <| Module k V


/-- The additive equivalence from the `Module (MonoidAlgebra k G)` to the original vector space
of the representative.

This is just the identity, but it is helpful for typechecking and keeping track of instances.
-/
def asModuleEquiv : ρ.asModule ≃+ V :=
  AddEquiv.refl _


@[simp]
theorem asModuleEquiv_map_smul (r : MonoidAlgebra k G) (x : ρ.asModule) :
    ρ.asModuleEquiv (r • x) = ρ.asAlgebraHom r (ρ.asModuleEquiv x) :=
  rfl


@[simp]
theorem asModuleEquiv_symm_map_smul (r : k) (x : V) :
    ρ.asModuleEquiv.symm (r • x) = algebraMap k (MonoidAlgebra k G) r • ρ.asModuleEquiv.symm x := by
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    inst✝³ : CommSemiring k
    inst✝² : Monoid G
    inst✝¹ : AddCommMonoid V
    inst✝ : Module k V
    ρ : Representation k G V
    r : k
    x : V
    ⊢ Eq (ρ.asModuleEquiv.symm (HSMul.hSMul r x)) (HSMul.hSMul ((algebraMap k (Mon …
  -/
  apply_fun ρ.asModuleEquiv
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    inst✝³ : CommSemiring k
    inst✝² : Monoid G
    inst✝¹ : AddCommMonoid V
    inst✝ : Module k V
    ρ : Representation k G V
    r : k
    x : V
    ⊢ Eq (ρ.asModuleEquiv (ρ.asModuleEquiv.symm (HSMul.hSMul r x))) (ρ.asModuleEqu …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem asModuleEquiv_symm_map_rho (g : G) (x : V) :
    ρ.asModuleEquiv.symm (ρ g x) = MonoidAlgebra.of k G g • ρ.asModuleEquiv.symm x := by
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    inst✝³ : CommSemiring k
    inst✝² : Monoid G
    inst✝¹ : AddCommMonoid V
    inst✝ : Module k V
    ρ : Representation k G V
    g : G
    x : V
    ⊢ Eq (ρ.asModuleEquiv.symm ((ρ g) x)) (HSMul.hSMul ((MonoidAlgebra.of k G) g)  …
  -/
  apply_fun ρ.asModuleEquiv
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    inst✝³ : CommSemiring k
    inst✝² : Monoid G
    inst✝¹ : AddCommMonoid V
    inst✝ : Module k V
    ρ : Representation k G V
    g : G
    x : V
    ⊢ Eq (ρ.asModuleEquiv (ρ.asModuleEquiv.symm ((ρ g) x))) (ρ.asModuleEquiv (HSMu …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Build a `Representation k G M` from a `[Module (MonoidAlgebra k G) M]`.

This version is not always what we want, as it relies on an existing `[Module k M]`
instance, along with a `[IsScalarTower k (MonoidAlgebra k G) M]` instance.

We remedy this below in `ofModule`
(with the tradeoff that the representation is defined
only on a type synonym of the original module.)
-/
noncomputable def ofModule' (M : Type*) [AddCommMonoid M] [Module k M]
    [Module (MonoidAlgebra k G) M] [IsScalarTower k (MonoidAlgebra k G) M] : Representation k G M :=
  (MonoidAlgebra.lift k G (M →ₗ[k] M)).symm (Algebra.lsmul k k M)


/-- Build a `Representation` from a `[Module (MonoidAlgebra k G) M]`.

Note that the representation is built on `restrictScalars k (MonoidAlgebra k G) M`,
rather than on `M` itself.
-/
noncomputable def ofModule : Representation k G (RestrictScalars k (MonoidAlgebra k G) M) :=
  (MonoidAlgebra.lift k G
        (RestrictScalars k (MonoidAlgebra k G) M →ₗ[k]
          RestrictScalars k (MonoidAlgebra k G) M)).symm
    (RestrictScalars.lsmul k (MonoidAlgebra k G) M)


@[simp]
theorem ofModule_asAlgebraHom_apply_apply (r : MonoidAlgebra k G)
    (m : RestrictScalars k (MonoidAlgebra k G) M) :
    ((ofModule M).asAlgebraHom r) m =
      (RestrictScalars.addEquiv _ _ _).symm (r • RestrictScalars.addEquiv _ _ _ m) := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝³ : CommSemiring k
    inst✝² : Monoid G
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : Module (MonoidAlgebra k G) M
    r : MonoidAlgebra k G
    m : RestrictScalars k (MonoidAlgebra k G) M
    ⊢ Eq (((Representation.ofModule M).asAlgebraHom r) m) ((RestrictScalars.addEqu …
  -/
  apply MonoidAlgebra.induction_on r
    /-
      case hM
      k : Type u_1
      G : Type u_2
      inst✝³ : CommSemiring k
      inst✝² : Monoid G
      M : Type u_4
      inst✝¹ : AddCommMonoid M
      inst✝ : Module (MonoidAlgebra k G) M
      r : MonoidAlgebra k G
      m : RestrictScalars k (MonoidAlgebra k G) M
      ⊢ ∀ (g : G), Eq (((Representation.ofModule M).asAlgebraHom ((MonoidAlgebra.of  …
    -/
  · intro g
    simp only [one_smul, MonoidAlgebra.lift_symm_apply, MonoidAlgebra.of_apply,
      Representation.asAlgebraHom_single, Representation.ofModule, AddEquiv.apply_eq_iff_eq,
      RestrictScalars.lsmul_apply_apply]
    /-
      case hadd
      k : Type u_1
      G : Type u_2
      inst✝³ : CommSemiring k
      inst✝² : Monoid G
      M : Type u_4
      inst✝¹ : AddCommMonoid M
      inst✝ : Module (MonoidAlgebra k G) M
      r : MonoidAlgebra k G
      m : RestrictScalars k (MonoidAlgebra k G) M
      ⊢ ∀ (f g : MonoidAlgebra k G), Eq (((Representation.ofModule M).asAlgebraHom f …
    -/
  · intro f g fw gw
    /-
      case hadd
      k : Type u_1
      G : Type u_2
      inst✝³ : CommSemiring k
      inst✝² : Monoid G
      M : Type u_4
      inst✝¹ : AddCommMonoid M
      inst✝ : Module (MonoidAlgebra k G) M
      r : MonoidAlgebra k G
      m : RestrictScalars k (MonoidAlgebra k G) M
      f g : MonoidAlgebra k G
      fw : Eq (((Representation.ofModule M).asAlgebraHom f) m) ((RestrictScalars.add …
      gw : Eq (((Representation.ofModule M).asAlgebraHom g) m) ((RestrictScalars.add …
      ⊢ Eq (((Representation.ofModule M).asAlgebraHom (HAdd.hAdd f g)) m) ((Restrict …
    -/
    simp only [fw, gw, map_add, add_smul, LinearMap.add_apply]
    /-
      🎉 no goals
    -/
    /-
      case hsmul
      k : Type u_1
      G : Type u_2
      inst✝³ : CommSemiring k
      inst✝² : Monoid G
      M : Type u_4
      inst✝¹ : AddCommMonoid M
      inst✝ : Module (MonoidAlgebra k G) M
      r : MonoidAlgebra k G
      m : RestrictScalars k (MonoidAlgebra k G) M
      ⊢ ∀ (r : k) (f : MonoidAlgebra k G), Eq (((Representation.ofModule M).asAlgebr …
    -/
  · intro r f w
    /-
      case hsmul
      k : Type u_1
      G : Type u_2
      inst✝³ : CommSemiring k
      inst✝² : Monoid G
      M : Type u_4
      inst✝¹ : AddCommMonoid M
      inst✝ : Module (MonoidAlgebra k G) M
      r✝ : MonoidAlgebra k G
      m : RestrictScalars k (MonoidAlgebra k G) M
      r : k
      f : MonoidAlgebra k G
      w : Eq (((Representation.ofModule M).asAlgebraHom f) m) ((RestrictScalars.addE …
      ⊢ Eq (((Representation.ofModule M).asAlgebraHom (HSMul.hSMul r f)) m) ((Restri …
    -/
    simp only [w, map_smul, LinearMap.smul_apply, RestrictScalars.addEquiv_symm_map_smul_smul]
    /-
      🎉 no goals
    -/


@[simp]
theorem ofModule_asModule_act (g : G) (x : RestrictScalars k (MonoidAlgebra k G) ρ.asModule) :
    ofModule (k := k) (G := G) ρ.asModule g x = -- Porting note: more help with implicit
      (RestrictScalars.addEquiv _ _ _).symm
        (ρ.asModuleEquiv.symm (ρ g (ρ.asModuleEquiv (RestrictScalars.addEquiv _ _ _ x)))) := by
  apply_fun RestrictScalars.addEquiv _ _ ρ.asModule using
    (RestrictScalars.addEquiv _ _ ρ.asModule).injective
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    inst✝³ : CommSemiring k
    inst✝² : Monoid G
    inst✝¹ : AddCommMonoid V
    inst✝ : Module k V
    ρ : Representation k G V
    g : G
    x : RestrictScalars k (MonoidAlgebra k G) ρ.asModule
    ⊢ Eq ((RestrictScalars.addEquiv k (MonoidAlgebra k G) ρ.asModule) (((Represent …
  -/
  dsimp [ofModule, RestrictScalars.lsmul_apply_apply]
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    inst✝³ : CommSemiring k
    inst✝² : Monoid G
    inst✝¹ : AddCommMonoid V
    inst✝ : Module k V
    ρ : Representation k G V
    g : G
    x : RestrictScalars k (MonoidAlgebra k G) ρ.asModule
    ⊢ Eq ((RestrictScalars.addEquiv k (MonoidAlgebra k G) ρ.asModule) ((RestrictSc …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem smul_ofModule_asModule (r : MonoidAlgebra k G) (m : (ofModule M).asModule) :
    (RestrictScalars.addEquiv k _ _) ((ofModule M).asModuleEquiv (r • m)) =
      r • (RestrictScalars.addEquiv k _ _) ((ofModule M).asModuleEquiv (G := G) m) := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝³ : CommSemiring k
    inst✝² : Monoid G
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : Module (MonoidAlgebra k G) M
    r : MonoidAlgebra k G
    m : (Representation.ofModule M).asModule
    ⊢ Eq ((RestrictScalars.addEquiv k (MonoidAlgebra k G) M) ((Representation.ofMo …
  -/
  dsimp
  /-
    k : Type u_1
    G : Type u_2
    inst✝³ : CommSemiring k
    inst✝² : Monoid G
    M : Type u_4
    inst✝¹ : AddCommMonoid M
    inst✝ : Module (MonoidAlgebra k G) M
    r : MonoidAlgebra k G
    m : (Representation.ofModule M).asModule
    ⊢ Eq ((RestrictScalars.addEquiv k (MonoidAlgebra k G) M) (((Representation.ofM …
  -/
  simp only [AddEquiv.apply_symm_apply, ofModule_asAlgebraHom_apply_apply]
  /-
    🎉 no goals
  -/


instance : AddCommGroup ρ.asModule :=
  I


/-- A `G`-action on `H` induces a representation `G →* End(k[H])` in the natural way. -/
noncomputable def ofMulAction : Representation k G (H →₀ k) where
  toFun g := Finsupp.lmapDomain k k (g • ·)
  map_one' := by
    /-
      k : Type u_1
      inst✝² : CommSemiring k
      G : Type u_2
      inst✝¹ : Monoid G
      H : Type u_3
      inst✝ : MulAction G H
      ⊢ Eq ((fun g => Finsupp.lmapDomain k k fun x => HSMul.hSMul g x) 1) 1
    -/
    ext x y
    /-
      case h.h.h
      k : Type u_1
      inst✝² : CommSemiring k
      G : Type u_2
      inst✝¹ : Monoid G
      H : Type u_3
      inst✝ : MulAction G H
      x y : H
      ⊢ Eq (((((fun g => Finsupp.lmapDomain k k fun x => HSMul.hSMul g x) 1).comp (F …
    -/
    dsimp
    /-
      case h.h.h
      k : Type u_1
      inst✝² : CommSemiring k
      G : Type u_2
      inst✝¹ : Monoid G
      H : Type u_3
      inst✝ : MulAction G H
      x y : H
      ⊢ Eq ((Finsupp.mapDomain (fun x => HSMul.hSMul 1 x) (Finsupp.single x 1)) y) ( …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_mul' x y := by
    /-
      k : Type u_1
      inst✝² : CommSemiring k
      G : Type u_2
      inst✝¹ : Monoid G
      H : Type u_3
      inst✝ : MulAction G H
      x y : G
      ⊢ Eq ({ toFun := fun g => Finsupp.lmapDomain k k fun x => HSMul.hSMul g x, map …
    -/
    ext z w
    /-
      case h.h.h
      k : Type u_1
      inst✝² : CommSemiring k
      G : Type u_2
      inst✝¹ : Monoid G
      H : Type u_3
      inst✝ : MulAction G H
      x y : G
      z w : H
      ⊢ Eq (((({ toFun := fun g => Finsupp.lmapDomain k k fun x => HSMul.hSMul g x,  …
    -/
    simp [mul_smul]
    /-
      🎉 no goals
    -/


theorem ofMulAction_def (g : G) : ofMulAction k G H g = Finsupp.lmapDomain k k (g • ·) :=
  rfl


theorem ofMulAction_single (g : G) (x : H) (r : k) :
    ofMulAction k G H g (Finsupp.single x r) = Finsupp.single (g • x) r :=
  Finsupp.mapDomain_single


/-- Turns a `k`-module `A` with a compatible `DistribMulAction` of a monoid `G` into a
`k`-linear `G`-representation on `A`. -/
def ofDistribMulAction : Representation k G A where
  toFun := fun m =>
    { DistribMulAction.toAddMonoidEnd G A m with
      map_smul' := smul_comm _ }
                 /-
                   k : Type u_1
                   G : Type u_2
                   A : Type u_3
                   inst✝⁵ : CommSemiring k
                   inst✝⁴ : Monoid G
                   inst✝³ : AddCommMonoid A
                   inst✝² : Module k A
                   inst✝¹ : DistribMulAction G A
                   inst✝ : SMulCommClass G k A
                   ⊢ Eq
                       ((fun m =>
                           let __src := (DistribMulAction.toAddMonoidEnd G A) m;
                           { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ })
                         1)
                       1
                 -/
  map_one' := by ext; exact one_smul _ _
                      /-
                        🎉 no goals
                      -/
                 /-
                   k : Type u_1
                   G : Type u_2
                   A : Type u_3
                   inst✝⁵ : CommSemiring k
                   inst✝⁴ : Monoid G
                   inst✝³ : AddCommMonoid A
                   inst✝² : Module k A
                   inst✝¹ : DistribMulAction G A
                   inst✝ : SMulCommClass G k A
                   ⊢ ∀ (x y : G),
                       Eq
                         ({
                               toFun := fun m =>
                                 let __src := (DistribMulAction.toAddMonoidEnd G A) m;
                                 { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ },
                               map_one' := ⋯ }.toFun
                           (HMul.hMul x y))
                         (HMul.hMul
                           ({
                                 toFun := fun m =>
                                   let __src := (DistribMulAction.toAddMonoidEnd G A) m;
                                   { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ },
                                 map_one' := ⋯ }.toFun
                             x)
                           ({
                                 toFun := fun m =>
                                   let __src := (DistribMulAction.toAddMonoidEnd G A) m;
                                   { toFun := (↑__src).toFun, map_add' := ⋯, map_smul' := ⋯ },
                                 map_one' := ⋯ }.toFun
                             y))
                 -/
  map_mul' := by intros; ext; exact mul_smul _ _ _
                              /-
                                🎉 no goals
                              -/


@[simp] theorem ofDistribMulAction_apply_apply (g : G) (a : A) :
    ofDistribMulAction k G A g a = g • a := rfl


/-- Turns a `CommGroup` `G` with a `MulDistribMulAction` of a monoid `M` into a
`ℤ`-linear `M`-representation on `Additive G`. -/
def ofMulDistribMulAction : Representation ℤ M (Additive G) :=
  (addMonoidEndRingEquivInt (Additive G) : AddMonoid.End (Additive G) →* _).comp
    ((monoidEndToAdditive G : _ →* _).comp (MulDistribMulAction.toMonoidEnd M G))


@[simp] theorem ofMulDistribMulAction_apply_apply (g : M) (a : Additive G) :
    ofMulDistribMulAction M G g a = Additive.ofMul (g • a.toMul) := rfl


@[simp]
theorem ofMulAction_apply {H : Type*} [MulAction G H] (g : G) (f : H →₀ k) (h : H) :
    ofMulAction k G H g f h = f (g⁻¹ • h) := by
  /-
    k : Type u_1
    G : Type u_2
    inst✝² : CommSemiring k
    inst✝¹ : Group G
    H : Type u_4
    inst✝ : MulAction G H
    g : G
    f : Finsupp H k
    h : H
    ⊢ Eq ((((Representation.ofMulAction k G H) g) f) h) (f (HSMul.hSMul (Inv.inv g …
  -/
  conv_lhs => rw [← smul_inv_smul g h]
  /-
    k : Type u_1
    G : Type u_2
    inst✝² : CommSemiring k
    inst✝¹ : Group G
    H : Type u_4
    inst✝ : MulAction G H
    g : G
    f : Finsupp H k
    h : H
    ⊢ Eq ((((Representation.ofMulAction k G H) g) f) (HSMul.hSMul g (HSMul.hSMul ( …
  -/
  let h' := g⁻¹ • h
  /-
    k : Type u_1
    G : Type u_2
    inst✝² : CommSemiring k
    inst✝¹ : Group G
    H : Type u_4
    inst✝ : MulAction G H
    g : G
    f : Finsupp H k
    h : H
    h' : H := HSMul.hSMul (Inv.inv g) h
    ⊢ Eq ((((Representation.ofMulAction k G H) g) f) (HSMul.hSMul g (HSMul.hSMul ( …
  -/
  change ofMulAction k G H g f (g • h') = f h'
  have hg : Function.Injective (g • · : H → H) := by
    intro h₁ h₂
    simp
  /-
    k : Type u_1
    G : Type u_2
    inst✝² : CommSemiring k
    inst✝¹ : Group G
    H : Type u_4
    inst✝ : MulAction G H
    g : G
    f : Finsupp H k
    h : H
    h' : H := HSMul.hSMul (Inv.inv g) h
    hg : Function.Injective fun x => HSMul.hSMul g x
    ⊢ Eq ((((Representation.ofMulAction k G H) g) f) (HSMul.hSMul g h')) (f h')
  -/
  simp only [ofMulAction_def, Finsupp.lmapDomain_apply, Finsupp.mapDomain_apply, hg]
  /-
    🎉 no goals
  -/

-- Porting note: did not need this in ML3; noncomputable because IR check complains

noncomputable instance :
    HMul (MonoidAlgebra k G) ((ofMulAction k G G).asModule) (MonoidAlgebra k G) :=
  inferInstanceAs <| HMul (MonoidAlgebra k G) (MonoidAlgebra k G) (MonoidAlgebra k G)


theorem ofMulAction_self_smul_eq_mul (x : MonoidAlgebra k G) (y : (ofMulAction k G G).asModule) :
    x • y = (x * y : MonoidAlgebra k G) := -- by
  -- Porting note: trouble figuring out the motive
  x.induction_on (p := fun z => z • y = z * y)
    (fun g => by
      /-
        k : Type u_1
        G : Type u_2
        inst✝¹ : CommSemiring k
        inst✝ : Group G
        x : MonoidAlgebra k G
        y : (Representation.ofMulAction k G G).asModule
        g : G
        ⊢ (fun z => Eq (HSMul.hSMul z y) (HMul.hMul z y)) ((MonoidAlgebra.of k G) g)
      -/
      show asAlgebraHom (ofMulAction k G G) _ _ = _; ext
      simp only [MonoidAlgebra.of_apply, asAlgebraHom_single, one_smul,
        ofMulAction_apply, smul_eq_mul]
      -- Porting note: single_mul_apply not firing in simp
      /-
        case h
        k : Type u_1
        G : Type u_2
        inst✝¹ : CommSemiring k
        inst✝ : Group G
        x : MonoidAlgebra k G
        y : (Representation.ofMulAction k G G).asModule
        g a✝ : G
        ⊢ Eq (y (HMul.hMul (Inv.inv g) a✝)) ((HMul.hMul (MonoidAlgebra.single g 1) y)  …
      -/
      rw [MonoidAlgebra.single_mul_apply, one_mul]
      /-
        🎉 no goals
      -/
    )
                         /-
                           k : Type u_1
                           G : Type u_2
                           inst✝¹ : CommSemiring k
                           inst✝ : Group G
                           x✝ : MonoidAlgebra k G
                           y✝ : (Representation.ofMulAction k G G).asModule
                           x y : MonoidAlgebra k G
                           hx : (fun z => Eq (HSMul.hSMul z y✝) (HMul.hMul z y✝)) x
                           hy : (fun z => Eq (HSMul.hSMul z y✝) (HMul.hMul z y✝)) y
                           ⊢ (fun z => Eq (HSMul.hSMul z y✝) (HMul.hMul z y✝)) (HAdd.hAdd x y)
                         -/
    (fun x y hx hy => by simp only [hx, hy, add_mul, add_smul]) fun r x hx => by
                         /-
                           🎉 no goals
                         -/
    /-
      k : Type u_1
      G : Type u_2
      inst✝¹ : CommSemiring k
      inst✝ : Group G
      x✝ : MonoidAlgebra k G
      y : (Representation.ofMulAction k G G).asModule
      r : k
      x : MonoidAlgebra k G
      hx : (fun z => Eq (HSMul.hSMul z y) (HMul.hMul z y)) x
      ⊢ (fun z => Eq (HSMul.hSMul z y) (HMul.hMul z y)) (HSMul.hSMul r x)
    -/
    show asAlgebraHom (ofMulAction k G G) _ _ = _  -- Porting note: was simpa [← hx]
    /-
      k : Type u_1
      G : Type u_2
      inst✝¹ : CommSemiring k
      inst✝ : Group G
      x✝ : MonoidAlgebra k G
      y : (Representation.ofMulAction k G G).asModule
      r : k
      x : MonoidAlgebra k G
      hx : (fun z => Eq (HSMul.hSMul z y) (HMul.hMul z y)) x
      ⊢ Eq (((Representation.ofMulAction k G G).asAlgebraHom (HSMul.hSMul r x)) y) ( …
    -/
    simp only [map_smul, smul_apply, Algebra.smul_mul_assoc]
    /-
      k : Type u_1
      G : Type u_2
      inst✝¹ : CommSemiring k
      inst✝ : Group G
      x✝ : MonoidAlgebra k G
      y : (Representation.ofMulAction k G G).asModule
      r : k
      x : MonoidAlgebra k G
      hx : (fun z => Eq (HSMul.hSMul z y) (HMul.hMul z y)) x
      ⊢ Eq (HSMul.hSMul r (((Representation.ofMulAction k G G).asAlgebraHom x) y)) ( …
    -/
    rw [← hx]
    /-
      k : Type u_1
      G : Type u_2
      inst✝¹ : CommSemiring k
      inst✝ : Group G
      x✝ : MonoidAlgebra k G
      y : (Representation.ofMulAction k G G).asModule
      r : k
      x : MonoidAlgebra k G
      hx : (fun z => Eq (HSMul.hSMul z y) (HMul.hMul z y)) x
      ⊢ Eq (HSMul.hSMul r (((Representation.ofMulAction k G G).asAlgebraHom x) y)) ( …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- If we equip `k[G]` with the `k`-linear `G`-representation induced by the left regular action of
`G` on itself, the resulting object is isomorphic as a `k[G]`-module to `k[G]` with its natural
`k[G]`-module structure. -/
@[simps]
noncomputable def ofMulActionSelfAsModuleEquiv :
    (ofMulAction k G G).asModule ≃ₗ[MonoidAlgebra k G] MonoidAlgebra k G :=
  { asModuleEquiv _ with map_smul' := ofMulAction_self_smul_eq_mul }


/-- When `G` is a group, a `k`-linear representation of `G` on `V` can be thought of as
a group homomorphism from `G` into the invertible `k`-linear endomorphisms of `V`.
-/
def asGroupHom : G →* Units (V →ₗ[k] V) :=
  MonoidHom.toHomUnits ρ


theorem asGroupHom_apply (g : G) : ↑(asGroupHom ρ g) = ρ g := by
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    inst✝³ : CommSemiring k
    inst✝² : Group G
    inst✝¹ : AddCommMonoid V
    inst✝ : Module k V
    ρ : Representation k G V
    g : G
    ⊢ Eq (↑(ρ.asGroupHom g)) (ρ g)
  -/
  simp only [asGroupHom, MonoidHom.coe_toHomUnits]
  /-
    🎉 no goals
  -/


/-- Given representations of `G` on `V` and `W`, there is a natural representation of `G` on their
tensor product `V ⊗[k] W`.
-/
noncomputable def tprod : Representation k G (V ⊗[k] W) where
  toFun g := TensorProduct.map (ρV g) (ρW g)
                 /-
                   k : Type u_1
                   G : Type u_2
                   V : Type u_3
                   W : Type u_4
                   inst✝⁵ : CommSemiring k
                   inst✝⁴ : Monoid G
                   inst✝³ : AddCommMonoid V
                   inst✝² : Module k V
                   inst✝¹ : AddCommMonoid W
                   inst✝ : Module k W
                   ρV : Representation k G V
                   ρW : Representation k G W
                   ⊢ Eq ((fun g => TensorProduct.map (ρV g) (ρW g)) 1) 1
                 -/
  map_one' := by simp only [map_one, TensorProduct.map_one]
                 /-
                   🎉 no goals
                 -/
                     /-
                       k : Type u_1
                       G : Type u_2
                       V : Type u_3
                       W : Type u_4
                       inst✝⁵ : CommSemiring k
                       inst✝⁴ : Monoid G
                       inst✝³ : AddCommMonoid V
                       inst✝² : Module k V
                       inst✝¹ : AddCommMonoid W
                       inst✝ : Module k W
                       ρV : Representation k G V
                       ρW : Representation k G W
                       g h : G
                       ⊢ Eq ({ toFun := fun g => TensorProduct.map (ρV g) (ρW g), map_one' := ⋯ }.toF …
                     -/
  map_mul' g h := by simp only [map_mul, TensorProduct.map_mul]
                     /-
                       🎉 no goals
                     -/


local notation ρV " ⊗ " ρW => tprod ρV ρW


@[simp]
theorem tprod_apply (g : G) : (ρV ⊗ ρW) g = TensorProduct.map (ρV g) (ρW g) :=
  rfl


theorem smul_tprod_one_asModule (r : MonoidAlgebra k G) (x : V) (y : W) :
    -- Porting note: required to since Lean 4 doesn't unfold asModule
    let x' : ρV.asModule := x
    let z : (ρV.tprod 1).asModule := x ⊗ₜ y
    r • z = (r • x') ⊗ₜ y := by
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁵ : CommSemiring k
    inst✝⁴ : Monoid G
    inst✝³ : AddCommMonoid V
    inst✝² : Module k V
    inst✝¹ : AddCommMonoid W
    inst✝ : Module k W
    ρV : Representation k G V
    r : MonoidAlgebra k G
    x : V
    y : W
    ⊢ let x' := x;
      let z := TensorProduct.tmul k x y;
      Eq (HSMul.hSMul r z) (TensorProduct.tmul k (HSMul.hSMul r x') y)
  -/
  show asAlgebraHom (ρV ⊗ 1) _ _ = asAlgebraHom ρV _ _ ⊗ₜ _
  simp only [asAlgebraHom_def, MonoidAlgebra.lift_apply, tprod_apply, MonoidHom.one_apply,
    LinearMap.finsupp_sum_apply, LinearMap.smul_apply, TensorProduct.map_tmul, LinearMap.one_apply]
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁵ : CommSemiring k
    inst✝⁴ : Monoid G
    inst✝³ : AddCommMonoid V
    inst✝² : Module k V
    inst✝¹ : AddCommMonoid W
    inst✝ : Module k W
    ρV : Representation k G V
    r : MonoidAlgebra k G
    x : V
    y : W
    ⊢ Eq (Finsupp.sum r fun i d => HSMul.hSMul d (TensorProduct.tmul k ((ρV i) x)  …
  -/
  simp only [Finsupp.sum, TensorProduct.sum_tmul]
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁵ : CommSemiring k
    inst✝⁴ : Monoid G
    inst✝³ : AddCommMonoid V
    inst✝² : Module k V
    inst✝¹ : AddCommMonoid W
    inst✝ : Module k W
    ρV : Representation k G V
    r : MonoidAlgebra k G
    x : V
    y : W
    ⊢ Eq (r.support.sum fun x_1 => HSMul.hSMul (r x_1) (TensorProduct.tmul k ((ρV  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem smul_one_tprod_asModule (r : MonoidAlgebra k G) (x : V) (y : W) :
    -- Porting note: required to since Lean 4 doesn't unfold asModule
    let y' : ρW.asModule := y
    let z : (1 ⊗ ρW).asModule := x ⊗ₜ y
    r • z = x ⊗ₜ (r • y') := by
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁵ : CommSemiring k
    inst✝⁴ : Monoid G
    inst✝³ : AddCommMonoid V
    inst✝² : Module k V
    inst✝¹ : AddCommMonoid W
    inst✝ : Module k W
    ρW : Representation k G W
    r : MonoidAlgebra k G
    x : V
    y : W
    ⊢ let y' := y;
      let z := TensorProduct.tmul k x y;
      Eq (HSMul.hSMul r z) (TensorProduct.tmul k x (HSMul.hSMul r y'))
  -/
  show asAlgebraHom (1 ⊗ ρW) _ _ = _ ⊗ₜ asAlgebraHom ρW _ _
  simp only [asAlgebraHom_def, MonoidAlgebra.lift_apply, tprod_apply, MonoidHom.one_apply,
    LinearMap.finsupp_sum_apply, LinearMap.smul_apply, TensorProduct.map_tmul, LinearMap.one_apply]
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁵ : CommSemiring k
    inst✝⁴ : Monoid G
    inst✝³ : AddCommMonoid V
    inst✝² : Module k V
    inst✝¹ : AddCommMonoid W
    inst✝ : Module k W
    ρW : Representation k G W
    r : MonoidAlgebra k G
    x : V
    y : W
    ⊢ Eq (Finsupp.sum r fun i d => HSMul.hSMul d (TensorProduct.tmul k x ((ρW i) y …
  -/
  simp only [Finsupp.sum, TensorProduct.tmul_sum, TensorProduct.tmul_smul]
  /-
    🎉 no goals
  -/


/-- Given representations of `G` on `V` and `W`, there is a natural representation of `G` on the
module `V →ₗ[k] W`, where `G` acts by conjugation.
-/
def linHom : Representation k G (V →ₗ[k] W) where
  toFun g :=
    { toFun := fun f => ρW g ∘ₗ f ∘ₗ ρV g⁻¹
                                  /-
                                    k : Type u_1
                                    G : Type u_2
                                    V : Type u_3
                                    W : Type u_4
                                    inst✝⁵ : CommSemiring k
                                    inst✝⁴ : Group G
                                    inst✝³ : AddCommMonoid V
                                    inst✝² : Module k V
                                    inst✝¹ : AddCommMonoid W
                                    inst✝ : Module k W
                                    ρV : Representation k G V
                                    ρW : Representation k G W
                                    g : G
                                    f₁ f₂ : LinearMap (RingHom.id k) V W
                                    ⊢ Eq ((fun f => (ρW g).comp (f.comp (ρV (Inv.inv g)))) (HAdd.hAdd f₁ f₂)) (HAd …
                                  -/
      map_add' := fun f₁ f₂ => by simp_rw [add_comp, comp_add]
                                  /-
                                    🎉 no goals
                                  -/
                                 /-
                                   k : Type u_1
                                   G : Type u_2
                                   V : Type u_3
                                   W : Type u_4
                                   inst✝⁵ : CommSemiring k
                                   inst✝⁴ : Group G
                                   inst✝³ : AddCommMonoid V
                                   inst✝² : Module k V
                                   inst✝¹ : AddCommMonoid W
                                   inst✝ : Module k W
                                   ρV : Representation k G V
                                   ρW : Representation k G W
                                   g : G
                                   r : k
                                   f : LinearMap (RingHom.id k) V W
                                   ⊢ Eq ({ toFun := fun f => (ρW g).comp (f.comp (ρV (Inv.inv g))), map_add' := ⋯ …
                                 -/
      map_smul' := fun r f => by simp_rw [RingHom.id_apply, smul_comp, comp_smul] }
                                 /-
                                   🎉 no goals
                                 -/
  map_one' :=
    LinearMap.ext fun x => by
      /-
        k : Type u_1
        G : Type u_2
        V : Type u_3
        W : Type u_4
        inst✝⁵ : CommSemiring k
        inst✝⁴ : Group G
        inst✝³ : AddCommMonoid V
        inst✝² : Module k V
        inst✝¹ : AddCommMonoid W
        inst✝ : Module k W
        ρV : Representation k G V
        ρW : Representation k G W
        x : LinearMap (RingHom.id k) V W
        ⊢ Eq (((fun g => { toFun := fun f => (ρW g).comp (f.comp (ρV (Inv.inv g))), ma …
      -/
      dsimp -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11227):now needed
      /-
        k : Type u_1
        G : Type u_2
        V : Type u_3
        W : Type u_4
        inst✝⁵ : CommSemiring k
        inst✝⁴ : Group G
        inst✝³ : AddCommMonoid V
        inst✝² : Module k V
        inst✝¹ : AddCommMonoid W
        inst✝ : Module k W
        ρV : Representation k G V
        ρW : Representation k G W
        x : LinearMap (RingHom.id k) V W
        ⊢ Eq ((ρW 1).comp (x.comp (ρV (Inv.inv 1)))) x
      -/
      simp_rw [inv_one, map_one, one_eq_id, comp_id, id_comp]
      /-
        🎉 no goals
      -/
  map_mul' g h :=
    LinearMap.ext fun x => by
      /-
        k : Type u_1
        G : Type u_2
        V : Type u_3
        W : Type u_4
        inst✝⁵ : CommSemiring k
        inst✝⁴ : Group G
        inst✝³ : AddCommMonoid V
        inst✝² : Module k V
        inst✝¹ : AddCommMonoid W
        inst✝ : Module k W
        ρV : Representation k G V
        ρW : Representation k G W
        g h : G
        x : LinearMap (RingHom.id k) V W
        ⊢ Eq (({ toFun := fun g => { toFun := fun f => (ρW g).comp (f.comp (ρV (Inv.in …
      -/
      dsimp -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11227):now needed
      /-
        k : Type u_1
        G : Type u_2
        V : Type u_3
        W : Type u_4
        inst✝⁵ : CommSemiring k
        inst✝⁴ : Group G
        inst✝³ : AddCommMonoid V
        inst✝² : Module k V
        inst✝¹ : AddCommMonoid W
        inst✝ : Module k W
        ρV : Representation k G V
        ρW : Representation k G W
        g h : G
        x : LinearMap (RingHom.id k) V W
        ⊢ Eq ((ρW (HMul.hMul g h)).comp (x.comp (ρV (Inv.inv (HMul.hMul g h))))) ((ρW  …
      -/
      simp_rw [mul_inv_rev, map_mul, mul_eq_comp, comp_assoc]
      /-
        🎉 no goals
      -/


@[simp]
theorem linHom_apply (g : G) (f : V →ₗ[k] W) : (linHom ρV ρW) g f = ρW g ∘ₗ f ∘ₗ ρV g⁻¹ :=
  rfl


/-- The dual of a representation `ρ` of `G` on a module `V`, given by `(dual ρ) g f = f ∘ₗ (ρ g⁻¹)`,
where `f : Module.Dual k V`.
-/
def dual : Representation k G (Module.Dual k V) where
  toFun g :=
    { toFun := fun f => f ∘ₗ ρV g⁻¹
                                  /-
                                    k : Type u_1
                                    G : Type u_2
                                    V : Type u_3
                                    W : Type u_4
                                    inst✝⁵ : CommSemiring k
                                    inst✝⁴ : Group G
                                    inst✝³ : AddCommMonoid V
                                    inst✝² : Module k V
                                    inst✝¹ : AddCommMonoid W
                                    inst✝ : Module k W
                                    ρV : Representation k G V
                                    ρW : Representation k G W
                                    g : G
                                    f₁ f₂ : Module.Dual k V
                                    ⊢ Eq ((fun f => LinearMap.comp f (ρV (Inv.inv g))) (HAdd.hAdd f₁ f₂)) (HAdd.hA …
                                  -/
      map_add' := fun f₁ f₂ => by simp only [add_comp]
                                  /-
                                    🎉 no goals
                                  -/
      map_smul' := fun r f => by
        /-
          k : Type u_1
          G : Type u_2
          V : Type u_3
          W : Type u_4
          inst✝⁵ : CommSemiring k
          inst✝⁴ : Group G
          inst✝³ : AddCommMonoid V
          inst✝² : Module k V
          inst✝¹ : AddCommMonoid W
          inst✝ : Module k W
          ρV : Representation k G V
          ρW : Representation k G W
          g : G
          r : k
          f : Module.Dual k V
          ⊢ Eq ({ toFun := fun f => LinearMap.comp f (ρV (Inv.inv g)), map_add' := ⋯ }.t …
        -/
        ext
        /-
          case h
          k : Type u_1
          G : Type u_2
          V : Type u_3
          W : Type u_4
          inst✝⁵ : CommSemiring k
          inst✝⁴ : Group G
          inst✝³ : AddCommMonoid V
          inst✝² : Module k V
          inst✝¹ : AddCommMonoid W
          inst✝ : Module k W
          ρV : Representation k G V
          ρW : Representation k G W
          g : G
          r : k
          f : Module.Dual k V
          x✝ : V
          ⊢ Eq (({ toFun := fun f => LinearMap.comp f (ρV (Inv.inv g)), map_add' := ⋯ }. …
        -/
        simp only [coe_comp, Function.comp_apply, smul_apply, RingHom.id_apply] }
        /-
          🎉 no goals
        -/
  map_one' := by
    /-
      k : Type u_1
      G : Type u_2
      V : Type u_3
      W : Type u_4
      inst✝⁵ : CommSemiring k
      inst✝⁴ : Group G
      inst✝³ : AddCommMonoid V
      inst✝² : Module k V
      inst✝¹ : AddCommMonoid W
      inst✝ : Module k W
      ρV : Representation k G V
      ρW : Representation k G W
      ⊢ Eq ((fun g => { toFun := fun f => LinearMap.comp f (ρV (Inv.inv g)), map_add …
    -/
    ext
    /-
      case h.h
      k : Type u_1
      G : Type u_2
      V : Type u_3
      W : Type u_4
      inst✝⁵ : CommSemiring k
      inst✝⁴ : Group G
      inst✝³ : AddCommMonoid V
      inst✝² : Module k V
      inst✝¹ : AddCommMonoid W
      inst✝ : Module k W
      ρV : Representation k G V
      ρW : Representation k G W
      x✝¹ : Module.Dual k V
      x✝ : V
      ⊢ Eq ((((fun g => { toFun := fun f => LinearMap.comp f (ρV (Inv.inv g)), map_a …
    -/
    dsimp -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11227):now needed
    /-
      case h.h
      k : Type u_1
      G : Type u_2
      V : Type u_3
      W : Type u_4
      inst✝⁵ : CommSemiring k
      inst✝⁴ : Group G
      inst✝³ : AddCommMonoid V
      inst✝² : Module k V
      inst✝¹ : AddCommMonoid W
      inst✝ : Module k W
      ρV : Representation k G V
      ρW : Representation k G W
      x✝¹ : Module.Dual k V
      x✝ : V
      ⊢ Eq (x✝¹ ((ρV (Inv.inv 1)) x✝)) (x✝¹ x✝)
    -/
    simp only [coe_comp, Function.comp_apply, map_one, inv_one, coe_mk, one_apply]
    /-
      🎉 no goals
    -/
  map_mul' g h := by
    /-
      k : Type u_1
      G : Type u_2
      V : Type u_3
      W : Type u_4
      inst✝⁵ : CommSemiring k
      inst✝⁴ : Group G
      inst✝³ : AddCommMonoid V
      inst✝² : Module k V
      inst✝¹ : AddCommMonoid W
      inst✝ : Module k W
      ρV : Representation k G V
      ρW : Representation k G W
      g h : G
      ⊢ Eq ({ toFun := fun g => { toFun := fun f => LinearMap.comp f (ρV (Inv.inv g) …
    -/
    ext
    /-
      case h.h
      k : Type u_1
      G : Type u_2
      V : Type u_3
      W : Type u_4
      inst✝⁵ : CommSemiring k
      inst✝⁴ : Group G
      inst✝³ : AddCommMonoid V
      inst✝² : Module k V
      inst✝¹ : AddCommMonoid W
      inst✝ : Module k W
      ρV : Representation k G V
      ρW : Representation k G W
      g h : G
      x✝¹ : Module.Dual k V
      x✝ : V
      ⊢ Eq ((({ toFun := fun g => { toFun := fun f => LinearMap.comp f (ρV (Inv.inv  …
    -/
    dsimp -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11227):now needed
    /-
      case h.h
      k : Type u_1
      G : Type u_2
      V : Type u_3
      W : Type u_4
      inst✝⁵ : CommSemiring k
      inst✝⁴ : Group G
      inst✝³ : AddCommMonoid V
      inst✝² : Module k V
      inst✝¹ : AddCommMonoid W
      inst✝ : Module k W
      ρV : Representation k G V
      ρW : Representation k G W
      g h : G
      x✝¹ : Module.Dual k V
      x✝ : V
      ⊢ Eq (x✝¹ ((ρV (Inv.inv (HMul.hMul g h))) x✝)) (x✝¹ ((ρV (Inv.inv h)) ((ρV (In …
    -/
    simp only [coe_comp, Function.comp_apply, mul_inv_rev, map_mul, coe_mk, mul_apply]
    /-
      🎉 no goals
    -/


@[simp]
theorem dual_apply (g : G) : (dual ρV) g = Module.Dual.transpose (R := k) (ρV g⁻¹) :=
  rfl


/-- Given $k$-modules $V, W$, there is a homomorphism $φ : V^* ⊗ W → Hom_k(V, W)$
(implemented by `dualTensorHom` in `Mathlib.LinearAlgebra.Contraction`).
Given representations of $G$ on $V$ and $W$,there are representations of $G$ on $V^* ⊗ W$ and on
$Hom_k(V, W)$.
This lemma says that $φ$ is $G$-linear.
-/
theorem dualTensorHom_comm (g : G) :
    dualTensorHom k V W ∘ₗ TensorProduct.map (ρV.dual g) (ρW g) =
      (linHom ρV ρW) g ∘ₗ dualTensorHom k V W := by
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    W : Type u_4
    inst✝⁵ : CommSemiring k
    inst✝⁴ : Group G
    inst✝³ : AddCommMonoid V
    inst✝² : Module k V
    inst✝¹ : AddCommMonoid W
    inst✝ : Module k W
    ρV : Representation k G V
    ρW : Representation k G W
    g : G
    ⊢ Eq ((dualTensorHom k V W).comp (TensorProduct.map (ρV.dual g) (ρW g))) (((ρV …
  -/
  ext; simp [Module.Dual.transpose_apply]
       /-
         🎉 no goals
       -/


