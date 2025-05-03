/-- A non_unital `k`-algebra homomorphism from `MonoidAlgebra k G` is uniquely defined by its
values on the functions `single a 1`. -/
theorem nonUnitalAlgHom_ext [DistribMulAction k A] {φ₁ φ₂ : MonoidAlgebra k G →ₙₐ[k] A}
    (h : ∀ x, φ₁ (single x 1) = φ₂ (single x 1)) : φ₁ = φ₂ :=
  NonUnitalAlgHom.to_distribMulActionHom_injective <|
    Finsupp.distribMulActionHom_ext' fun a => DistribMulActionHom.ext_ring (h a)


/-- See note [partially-applied ext lemmas]. -/
@[ext high]
theorem nonUnitalAlgHom_ext' [DistribMulAction k A] {φ₁ φ₂ : MonoidAlgebra k G →ₙₐ[k] A}
    (h : φ₁.toMulHom.comp (ofMagma k G) = φ₂.toMulHom.comp (ofMagma k G)) : φ₁ = φ₂ :=
  nonUnitalAlgHom_ext k <| DFunLike.congr_fun h


/-- The functor `G ↦ MonoidAlgebra k G`, from the category of magmas to the category of non-unital,
non-associative algebras over `k` is adjoint to the forgetful functor in the other direction. -/
@[simps apply_apply symm_apply]
def liftMagma [Module k A] [IsScalarTower k A A] [SMulCommClass k A A] :
    (G →ₙ* A) ≃ (MonoidAlgebra k G →ₙₐ[k] A) where
  toFun f :=
    { liftAddHom fun x => (smulAddHom k A).flip (f x) with
      toFun := fun a => a.sum fun m t => t • f m
      map_smul' := fun t' a => by
        -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
        /-
          k : Type u₁
          G : Type u₂
          H : Type u_1
          R : Type u_2
          inst✝⁶ : Semiring k
          inst✝⁵ : DistribSMul R k
          inst✝⁴ : Mul G
          A : Type u₃
          inst✝³ : NonUnitalNonAssocSemiring A
          inst✝² : Module k A
          inst✝¹ : IsScalarTower k A A
          inst✝ : SMulCommClass k A A
          f : MulHom G A
          t' : k
          a : MonoidAlgebra k G
          ⊢ Eq ((fun a => Finsupp.sum a fun m t => HSMul.hSMul t (f m)) (HSMul.hSMul t'  …
        -/
        beta_reduce
        /-
          k : Type u₁
          G : Type u₂
          H : Type u_1
          R : Type u_2
          inst✝⁶ : Semiring k
          inst✝⁵ : DistribSMul R k
          inst✝⁴ : Mul G
          A : Type u₃
          inst✝³ : NonUnitalNonAssocSemiring A
          inst✝² : Module k A
          inst✝¹ : IsScalarTower k A A
          inst✝ : SMulCommClass k A A
          f : MulHom G A
          t' : k
          a : MonoidAlgebra k G
          ⊢ Eq (Finsupp.sum (HSMul.hSMul t' a) fun m t => HSMul.hSMul t (f m)) (HSMul.hS …
        -/
        rw [Finsupp.smul_sum, sum_smul_index']
          /-
            k : Type u₁
            G : Type u₂
            H : Type u_1
            R : Type u_2
            inst✝⁶ : Semiring k
            inst✝⁵ : DistribSMul R k
            inst✝⁴ : Mul G
            A : Type u₃
            inst✝³ : NonUnitalNonAssocSemiring A
            inst✝² : Module k A
            inst✝¹ : IsScalarTower k A A
            inst✝ : SMulCommClass k A A
            f : MulHom G A
            t' : k
            a : MonoidAlgebra k G
            ⊢ Eq (Finsupp.sum a fun i c => HSMul.hSMul (HSMul.hSMul t' c) (f i)) (Finsupp. …
          -/
        · simp_rw [smul_assoc, MonoidHom.id_apply]
          /-
            🎉 no goals
          -/
          /-
            k : Type u₁
            G : Type u₂
            H : Type u_1
            R : Type u_2
            inst✝⁶ : Semiring k
            inst✝⁵ : DistribSMul R k
            inst✝⁴ : Mul G
            A : Type u₃
            inst✝³ : NonUnitalNonAssocSemiring A
            inst✝² : Module k A
            inst✝¹ : IsScalarTower k A A
            inst✝ : SMulCommClass k A A
            f : MulHom G A
            t' : k
            a : MonoidAlgebra k G
            ⊢ ∀ (i : G), Eq (HSMul.hSMul 0 (f i)) 0
          -/
        · intro m
          /-
            k : Type u₁
            G : Type u₂
            H : Type u_1
            R : Type u_2
            inst✝⁶ : Semiring k
            inst✝⁵ : DistribSMul R k
            inst✝⁴ : Mul G
            A : Type u₃
            inst✝³ : NonUnitalNonAssocSemiring A
            inst✝² : Module k A
            inst✝¹ : IsScalarTower k A A
            inst✝ : SMulCommClass k A A
            f : MulHom G A
            t' : k
            a : MonoidAlgebra k G
            m : G
            ⊢ Eq (HSMul.hSMul 0 (f m)) 0
          -/
          exact zero_smul k (f m)
          /-
            🎉 no goals
          -/
      map_mul' := fun a₁ a₂ => by
        /-
          k : Type u₁
          G : Type u₂
          H : Type u_1
          R : Type u_2
          inst✝⁶ : Semiring k
          inst✝⁵ : DistribSMul R k
          inst✝⁴ : Mul G
          A : Type u₃
          inst✝³ : NonUnitalNonAssocSemiring A
          inst✝² : Module k A
          inst✝¹ : IsScalarTower k A A
          inst✝ : SMulCommClass k A A
          f : MulHom G A
          a₁ a₂ : MonoidAlgebra k G
          ⊢ Eq ({ toFun := fun a => Finsupp.sum a fun m t => HSMul.hSMul t (f m), map_sm …
        -/
        let g : G → k → A := fun m t => t • f m
        have h₁ : ∀ m, g m 0 = 0 := by
          intro m
          exact zero_smul k (f m)
        have h₂ : ∀ (m) (t₁ t₂ : k), g m (t₁ + t₂) = g m t₁ + g m t₂ := by
          intros
          rw [← add_smul]
        -- Porting note: `reducible` cannot be `local` so proof gets long.
        simp_rw [Finsupp.mul_sum, Finsupp.sum_mul, smul_mul_smul_comm, ← f.map_mul, mul_def,
          sum_comm a₂ a₁]
        /-
          k : Type u₁
          G : Type u₂
          H : Type u_1
          R : Type u_2
          inst✝⁶ : Semiring k
          inst✝⁵ : DistribSMul R k
          inst✝⁴ : Mul G
          A : Type u₃
          inst✝³ : NonUnitalNonAssocSemiring A
          inst✝² : Module k A
          inst✝¹ : IsScalarTower k A A
          inst✝ : SMulCommClass k A A
          f : MulHom G A
          a₁ a₂ : MonoidAlgebra k G
          g : G → k → A := fun m t => HSMul.hSMul t (f m)
          h₁ : ∀ (m : G), Eq (g m 0) 0
          h₂ : ∀ (m : G) (t₁ t₂ : k), Eq (g m (HAdd.hAdd t₁ t₂)) (HAdd.hAdd (g m t₁) (g  …
          ⊢ Eq (Finsupp.sum (Finsupp.sum a₁ fun a₁ b₁ => Finsupp.sum a₂ fun a₂ b₂ => Mon …
        -/
        rw [sum_sum_index h₁ h₂]; congr; ext
        /-
          case e_g.h.h
          k : Type u₁
          G : Type u₂
          H : Type u_1
          R : Type u_2
          inst✝⁶ : Semiring k
          inst✝⁵ : DistribSMul R k
          inst✝⁴ : Mul G
          A : Type u₃
          inst✝³ : NonUnitalNonAssocSemiring A
          inst✝² : Module k A
          inst✝¹ : IsScalarTower k A A
          inst✝ : SMulCommClass k A A
          f : MulHom G A
          a₁ a₂ : MonoidAlgebra k G
          g : G → k → A := fun m t => HSMul.hSMul t (f m)
          h₁ : ∀ (m : G), Eq (g m 0) 0
          h₂ : ∀ (m : G) (t₁ t₂ : k), Eq (g m (HAdd.hAdd t₁ t₂)) (HAdd.hAdd (g m t₁) (g  …
          x✝¹ : G
          x✝ : k
          ⊢ Eq (Finsupp.sum (Finsupp.sum a₂ fun a₂ b₂ => MonoidAlgebra.single (HMul.hMul …
        -/
        rw [sum_sum_index h₁ h₂]; congr; ext
        /-
          case e_g.h.h.e_g.h.h
          k : Type u₁
          G : Type u₂
          H : Type u_1
          R : Type u_2
          inst✝⁶ : Semiring k
          inst✝⁵ : DistribSMul R k
          inst✝⁴ : Mul G
          A : Type u₃
          inst✝³ : NonUnitalNonAssocSemiring A
          inst✝² : Module k A
          inst✝¹ : IsScalarTower k A A
          inst✝ : SMulCommClass k A A
          f : MulHom G A
          a₁ a₂ : MonoidAlgebra k G
          g : G → k → A := fun m t => HSMul.hSMul t (f m)
          h₁ : ∀ (m : G), Eq (g m 0) 0
          h₂ : ∀ (m : G) (t₁ t₂ : k), Eq (g m (HAdd.hAdd t₁ t₂)) (HAdd.hAdd (g m t₁) (g  …
          x✝³ : G
          x✝² : k
          x✝¹ : G
          x✝ : k
          ⊢ Eq (Finsupp.sum (MonoidAlgebra.single (HMul.hMul x✝³ x✝¹) (HMul.hMul x✝² x✝) …
        -/
        rw [sum_single_index (h₁ _)] }
        /-
          🎉 no goals
        -/
  invFun F := F.toMulHom.comp (ofMagma k G)
  left_inv f := by
    /-
      k : Type u₁
      G : Type u₂
      H : Type u_1
      R : Type u_2
      inst✝⁶ : Semiring k
      inst✝⁵ : DistribSMul R k
      inst✝⁴ : Mul G
      A : Type u₃
      inst✝³ : NonUnitalNonAssocSemiring A
      inst✝² : Module k A
      inst✝¹ : IsScalarTower k A A
      inst✝ : SMulCommClass k A A
      f : MulHom G A
      ⊢ Eq
          ((fun F => F.toMulHom.comp (MonoidAlgebra.ofMagma k G))
            ((fun f =>
                let __src := Finsupp.liftAddHom fun x => (smulAddHom k A).flip (f x);
                { toFun := fun a => Finsupp.sum a fun m t => HSMul.hSMul t (f m), ma …
              f))
          f
    -/
    ext m
    simp only [NonUnitalAlgHom.coe_mk, ofMagma_apply, NonUnitalAlgHom.toMulHom_eq_coe,
      sum_single_index, Function.comp_apply, one_smul, zero_smul, MulHom.coe_comp,
      NonUnitalAlgHom.coe_to_mulHom]
  right_inv F := by
    /-
      k : Type u₁
      G : Type u₂
      H : Type u_1
      R : Type u_2
      inst✝⁶ : Semiring k
      inst✝⁵ : DistribSMul R k
      inst✝⁴ : Mul G
      A : Type u₃
      inst✝³ : NonUnitalNonAssocSemiring A
      inst✝² : Module k A
      inst✝¹ : IsScalarTower k A A
      inst✝ : SMulCommClass k A A
      F : NonUnitalAlgHom (MonoidHom.id k) (MonoidAlgebra k G) A
      ⊢ Eq
          ((fun f =>
              let __src := Finsupp.liftAddHom fun x => (smulAddHom k A).flip (f x);
              { toFun := fun a => Finsupp.sum a fun m t => HSMul.hSMul t (f m), map_ …
            ((fun F => F.toMulHom.comp (MonoidAlgebra.ofMagma k G)) F))
          F
    -/
    ext m
    simp only [NonUnitalAlgHom.coe_mk, ofMagma_apply, NonUnitalAlgHom.toMulHom_eq_coe,
      sum_single_index, Function.comp_apply, one_smul, zero_smul, MulHom.coe_comp,
      NonUnitalAlgHom.coe_to_mulHom]


/-- The instance `Algebra k (MonoidAlgebra A G)` whenever we have `Algebra k A`.

In particular this provides the instance `Algebra k (MonoidAlgebra k G)`.
-/
instance algebra {A : Type*} [CommSemiring k] [Semiring A] [Algebra k A] [Monoid G] :
    Algebra k (MonoidAlgebra A G) :=
  { singleOneRingHom.comp (algebraMap k A) with
    smul_def' := fun r a => by
      /-
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝³ : CommSemiring k
        inst✝² : Semiring A
        inst✝¹ : Algebra k A
        inst✝ : Monoid G
        r : k
        a : MonoidAlgebra A G
        ⊢ Eq (HSMul.hSMul r a) (HMul.hMul (__src✝ r) a)
      -/
      ext
      -- Porting note: Newly required.
      /-
        case H
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝³ : CommSemiring k
        inst✝² : Semiring A
        inst✝¹ : Algebra k A
        inst✝ : Monoid G
        r : k
        a : MonoidAlgebra A G
        x✝ : G
        ⊢ Eq ((HSMul.hSMul r a) x✝) ((HMul.hMul (__src✝ r) a) x✝)
      -/
      rw [Finsupp.coe_smul]
      /-
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝³ : CommSemiring k
        inst✝² : Semiring A
        inst✝¹ : Algebra k A
        inst✝ : Monoid G
        r : k
        f : MonoidAlgebra A G
        ⊢ Eq (HMul.hMul (__src✝ r) f) (HMul.hMul f (__src✝ r))
      -/
      /-
        case H
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝³ : CommSemiring k
        inst✝² : Semiring A
        inst✝¹ : Algebra k A
        inst✝ : Monoid G
        r : k
        a : MonoidAlgebra A G
        x✝ : G
        ⊢ Eq (HSMul.hSMul r (⇑a) x✝) ((HMul.hMul (__src✝ r) a) x✝)
      -/
      /-
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝³ : CommSemiring k
        inst✝² : Semiring A
        inst✝¹ : Algebra k A
        inst✝ : Monoid G
        r : k
        f : MonoidAlgebra A G
        x✝ : G
        ⊢ Eq ((HMul.hMul (__src✝ r) f) x✝) ((HMul.hMul f (__src✝ r)) x✝)
      -/
      simp [single_one_mul_apply, Algebra.smul_def, Pi.smul_apply]
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
    commutes' := fun r f => by
      refine Finsupp.ext fun _ => ?_
      simp [single_one_mul_apply, mul_single_one_apply, Algebra.commutes] }


/-- `Finsupp.single 1` as an `AlgHom` -/
@[simps! apply]
def singleOneAlgHom {A : Type*} [CommSemiring k] [Semiring A] [Algebra k A] [Monoid G] :
    A →ₐ[k] MonoidAlgebra A G :=
  { singleOneRingHom with
    commutes' := fun r => by
      /-
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝³ : CommSemiring k
        inst✝² : Semiring A
        inst✝¹ : Algebra k A
        inst✝ : Monoid G
        r : k
        ⊢ Eq ((↑↑__src✝).toFun ((algebraMap k A) r)) ((algebraMap k (MonoidAlgebra A G …
      -/
      ext
      /-
        case H
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝³ : CommSemiring k
        inst✝² : Semiring A
        inst✝¹ : Algebra k A
        inst✝ : Monoid G
        r : k
        x✝ : G
        ⊢ Eq (((↑↑__src✝).toFun ((algebraMap k A) r)) x✝) (((algebraMap k (MonoidAlgeb …
      -/
      simp
      /-
        case H
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝³ : CommSemiring k
        inst✝² : Semiring A
        inst✝¹ : Algebra k A
        inst✝ : Monoid G
        r : k
        x✝ : G
        ⊢ Eq ((Finsupp.single 1 ((algebraMap k A) r)) x✝) (((algebraMap k (MonoidAlgeb …
      -/
      rfl }
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_algebraMap {A : Type*} [CommSemiring k] [Semiring A] [Algebra k A] [Monoid G] :
    ⇑(algebraMap k (MonoidAlgebra A G)) = single 1 ∘ algebraMap k A :=
  rfl


theorem single_eq_algebraMap_mul_of [CommSemiring k] [Monoid G] (a : G) (b : k) :
                                                                     /-
                                                                       k : Type u₁
                                                                       G : Type u₂
                                                                       inst✝¹ : CommSemiring k
                                                                       inst✝ : Monoid G
                                                                       a : G
                                                                       b : k
                                                                       ⊢ Eq (MonoidAlgebra.single a b) (HMul.hMul ((algebraMap k (MonoidAlgebra k G)) …
                                                                     -/
    single a b = algebraMap k (MonoidAlgebra k G) b * of k G a := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem single_algebraMap_eq_algebraMap_mul_of {A : Type*} [CommSemiring k] [Semiring A]
    [Algebra k A] [Monoid G] (a : G) (b : k) :
                                                                                      /-
                                                                                        k : Type u₁
                                                                                        G : Type u₂
                                                                                        A : Type u_3
                                                                                        inst✝³ : CommSemiring k
                                                                                        inst✝² : Semiring A
                                                                                        inst✝¹ : Algebra k A
                                                                                        inst✝ : Monoid G
                                                                                        a : G
                                                                                        b : k
                                                                                        ⊢ Eq (MonoidAlgebra.single a ((algebraMap k A) b)) (HMul.hMul ((algebraMap k ( …
                                                                                      -/
    single a (algebraMap k A b) = algebraMap k (MonoidAlgebra A G) b * of A G a := by simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


/-- `liftNCRingHom` as an `AlgHom`, for when `f` is an `AlgHom` -/
def liftNCAlgHom (f : A →ₐ[k] B) (g : G →* B) (h_comm : ∀ x y, Commute (f x) (g y)) :
    MonoidAlgebra A G →ₐ[k] B :=
  { liftNCRingHom (f : A →+* B) g h_comm with
                    /-
                      k : Type u₁
                      G : Type u₂
                      H : Type u_1
                      R : Type u_2
                      inst✝⁶ : CommSemiring k
                      inst✝⁵ : Monoid G
                      inst✝⁴ : Monoid H
                      A : Type u₃
                      inst✝³ : Semiring A
                      inst✝² : Algebra k A
                      B : Type u_3
                      inst✝¹ : Semiring B
                      inst✝ : Algebra k B
                      f : AlgHom k A B
                      g : MonoidHom G B
                      h_comm : ∀ (x : A) (y : G), Commute (f x) (g y)
                      ⊢ ∀ (r : k), Eq ((↑↑__src✝).toFun ((algebraMap k (MonoidAlgebra A G)) r)) ((al …
                    -/
    commutes' := by simp [liftNCRingHom] }
                    /-
                      🎉 no goals
                    -/


/-- A `k`-algebra homomorphism from `MonoidAlgebra k G` is uniquely defined by its
values on the functions `single a 1`. -/
theorem algHom_ext ⦃φ₁ φ₂ : MonoidAlgebra k G →ₐ[k] A⦄
    (h : ∀ x, φ₁ (single x 1) = φ₂ (single x 1)) : φ₁ = φ₂ :=
  AlgHom.toLinearMap_injective <| Finsupp.lhom_ext' fun a => LinearMap.ext_ring (h a)

-- Porting note: The priority must be `high`.

/-- See note [partially-applied ext lemmas]. -/
@[ext high]
theorem algHom_ext' ⦃φ₁ φ₂ : MonoidAlgebra k G →ₐ[k] A⦄
    (h :
      (φ₁ : MonoidAlgebra k G →* A).comp (of k G) = (φ₂ : MonoidAlgebra k G →* A).comp (of k G)) :
    φ₁ = φ₂ :=
  algHom_ext <| DFunLike.congr_fun h


/-- Any monoid homomorphism `G →* A` can be lifted to an algebra homomorphism
`MonoidAlgebra k G →ₐ[k] A`. -/
def lift : (G →* A) ≃ (MonoidAlgebra k G →ₐ[k] A) where
  invFun f := (f : MonoidAlgebra k G →* A).comp (of k G)
  toFun F := liftNCAlgHom (Algebra.ofId k A) F fun _ _ => Algebra.commutes _ _
  left_inv f := by
    /-
      k : Type u₁
      G : Type u₂
      H : Type u_1
      R : Type u_2
      inst✝⁶ : CommSemiring k
      inst✝⁵ : Monoid G
      inst✝⁴ : Monoid H
      A : Type u₃
      inst✝³ : Semiring A
      inst✝² : Algebra k A
      B : Type u_3
      inst✝¹ : Semiring B
      inst✝ : Algebra k B
      f : MonoidHom G A
      ⊢ Eq ((fun f => (↑f).comp (MonoidAlgebra.of k G)) ((fun F => MonoidAlgebra.lif …
    -/
    ext
    /-
      case h
      k : Type u₁
      G : Type u₂
      H : Type u_1
      R : Type u_2
      inst✝⁶ : CommSemiring k
      inst✝⁵ : Monoid G
      inst✝⁴ : Monoid H
      A : Type u₃
      inst✝³ : Semiring A
      inst✝² : Algebra k A
      B : Type u_3
      inst✝¹ : Semiring B
      inst✝ : Algebra k B
      f : MonoidHom G A
      x✝ : G
      ⊢ Eq (((fun f => (↑f).comp (MonoidAlgebra.of k G)) ((fun F => MonoidAlgebra.li …
    -/
    simp [liftNCAlgHom, liftNCRingHom]
    /-
      🎉 no goals
    -/
  right_inv F := by
    /-
      k : Type u₁
      G : Type u₂
      H : Type u_1
      R : Type u_2
      inst✝⁶ : CommSemiring k
      inst✝⁵ : Monoid G
      inst✝⁴ : Monoid H
      A : Type u₃
      inst✝³ : Semiring A
      inst✝² : Algebra k A
      B : Type u_3
      inst✝¹ : Semiring B
      inst✝ : Algebra k B
      F : AlgHom k (MonoidAlgebra k G) A
      ⊢ Eq ((fun F => MonoidAlgebra.liftNCAlgHom (Algebra.ofId k A) F ⋯) ((fun f =>  …
    -/
    ext
    /-
      case h.h
      k : Type u₁
      G : Type u₂
      H : Type u_1
      R : Type u_2
      inst✝⁶ : CommSemiring k
      inst✝⁵ : Monoid G
      inst✝⁴ : Monoid H
      A : Type u₃
      inst✝³ : Semiring A
      inst✝² : Algebra k A
      B : Type u_3
      inst✝¹ : Semiring B
      inst✝ : Algebra k B
      F : AlgHom k (MonoidAlgebra k G) A
      x✝ : G
      ⊢ Eq (((↑((fun F => MonoidAlgebra.liftNCAlgHom (Algebra.ofId k A) F ⋯) ((fun f …
    -/
    simp [liftNCAlgHom, liftNCRingHom]
    /-
      🎉 no goals
    -/


theorem lift_apply' (F : G →* A) (f : MonoidAlgebra k G) :
    lift k G A F f = f.sum fun a b => algebraMap k A b * F a :=
  rfl


theorem lift_apply (F : G →* A) (f : MonoidAlgebra k G) :
                                                    /-
                                                      k : Type u₁
                                                      G : Type u₂
                                                      inst✝³ : CommSemiring k
                                                      inst✝² : Monoid G
                                                      A : Type u₃
                                                      inst✝¹ : Semiring A
                                                      inst✝ : Algebra k A
                                                      F : MonoidHom G A
                                                      f : MonoidAlgebra k G
                                                      ⊢ Eq (((MonoidAlgebra.lift k G A) F) f) (Finsupp.sum f fun a b => HSMul.hSMul  …
                                                    -/
    lift k G A F f = f.sum fun a b => b • F a := by simp only [lift_apply', Algebra.smul_def]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem lift_def (F : G →* A) : ⇑(lift k G A F) = liftNC ((algebraMap k A : k →+* A) : k →+ A) F :=
  rfl


@[simp]
theorem lift_symm_apply (F : MonoidAlgebra k G →ₐ[k] A) (x : G) :
    (lift k G A).symm F x = F (single x 1) :=
  rfl


@[simp]
theorem lift_single (F : G →* A) (a b) : lift k G A F (single a b) = b • F a := by
  /-
    k : Type u₁
    G : Type u₂
    inst✝³ : CommSemiring k
    inst✝² : Monoid G
    A : Type u₃
    inst✝¹ : Semiring A
    inst✝ : Algebra k A
    F : MonoidHom G A
    a : G
    b : k
    ⊢ Eq (((MonoidAlgebra.lift k G A) F) (MonoidAlgebra.single a b)) (HSMul.hSMul  …
  -/
  rw [lift_def, liftNC_single, Algebra.smul_def, AddMonoidHom.coe_coe]
  /-
    🎉 no goals
  -/


                                                                       /-
                                                                         k : Type u₁
                                                                         G : Type u₂
                                                                         inst✝³ : CommSemiring k
                                                                         inst✝² : Monoid G
                                                                         A : Type u₃
                                                                         inst✝¹ : Semiring A
                                                                         inst✝ : Algebra k A
                                                                         F : MonoidHom G A
                                                                         x : G
                                                                         ⊢ Eq (((MonoidAlgebra.lift k G A) F) ((MonoidAlgebra.of k G) x)) (F x)
                                                                       -/
theorem lift_of (F : G →* A) (x) : lift k G A F (of k G x) = F x := by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem lift_unique' (F : MonoidAlgebra k G →ₐ[k] A) :
    F = lift k G A ((F : MonoidAlgebra k G →* A).comp (of k G)) :=
  ((lift k G A).apply_symm_apply F).symm


/-- Decomposition of a `k`-algebra homomorphism from `MonoidAlgebra k G` by
its values on `F (single a 1)`. -/
theorem lift_unique (F : MonoidAlgebra k G →ₐ[k] A) (f : MonoidAlgebra k G) :
    F f = f.sum fun a b => b • F (single a 1) := by
  conv_lhs =>
    rw [lift_unique' F]
    simp [lift_apply]


/-- If `f : G → H` is a homomorphism between two magmas, then
`Finsupp.mapDomain f` is a non-unital algebra homomorphism between their magma algebras. -/
@[simps apply]
def mapDomainNonUnitalAlgHom (k A : Type*) [CommSemiring k] [Semiring A] [Algebra k A]
    {G H F : Type*} [Mul G] [Mul H] [FunLike F G H] [MulHomClass F G H] (f : F) :
    MonoidAlgebra A G →ₙₐ[k] MonoidAlgebra A H :=
  { (Finsupp.mapDomain.addMonoidHom f : MonoidAlgebra A G →+ MonoidAlgebra A H) with
    map_mul' := fun x y => mapDomain_mul f x y
    map_smul' := fun r x => mapDomain_smul r x }


variable (A) in
theorem mapDomain_algebraMap {F : Type*} [FunLike F G H] [MonoidHomClass F G H] (f : F) (r : k) :
    mapDomain f (algebraMap k (MonoidAlgebra A G) r) = algebraMap k (MonoidAlgebra A H) r := by
  /-
    k : Type u₁
    G : Type u₂
    H : Type u_1
    inst✝⁶ : CommSemiring k
    inst✝⁵ : Monoid G
    inst✝⁴ : Monoid H
    A : Type u₃
    inst✝³ : Semiring A
    inst✝² : Algebra k A
    F : Type u_4
    inst✝¹ : FunLike F G H
    inst✝ : MonoidHomClass F G H
    f : F
    r : k
    ⊢ Eq (MonoidAlgebra.mapDomain (⇑f) ((algebraMap k (MonoidAlgebra A G)) r)) ((a …
  -/
  simp only [coe_algebraMap, mapDomain_single, map_one, (· ∘ ·)]
  /-
    🎉 no goals
  -/


/-- If `f : G → H` is a multiplicative homomorphism between two monoids, then
`Finsupp.mapDomain f` is an algebra homomorphism between their monoid algebras. -/
@[simps!]
def mapDomainAlgHom (k A : Type*) [CommSemiring k] [Semiring A] [Algebra k A] {H F : Type*}
    [Monoid H] [FunLike F G H] [MonoidHomClass F G H] (f : F) :
    MonoidAlgebra A G →ₐ[k] MonoidAlgebra A H :=
  { mapDomainRingHom A f with commutes' := mapDomain_algebraMap A f }


@[simp]
lemma mapDomainAlgHom_id (k A) [CommSemiring k] [Semiring A] [Algebra k A] :
    mapDomainAlgHom k A (MonoidHom.id G) = AlgHom.id k (MonoidAlgebra A G) := by
  /-
    G : Type u₂
    inst✝³ : Monoid G
    k : Type u_4
    A : Type u_5
    inst✝² : CommSemiring k
    inst✝¹ : Semiring A
    inst✝ : Algebra k A
    ⊢ Eq (MonoidAlgebra.mapDomainAlgHom k A (MonoidHom.id G)) (AlgHom.id k (Monoid …
  -/
  ext; simp [MonoidHom.id, ← Function.id_def]
       /-
         🎉 no goals
       -/


@[simp]
lemma mapDomainAlgHom_comp (k A) {G₁ G₂ G₃} [CommSemiring k] [Semiring A] [Algebra k A]
    [Monoid G₁] [Monoid G₂] [Monoid G₃] (f : G₁ →* G₂) (g : G₂ →* G₃) :
    mapDomainAlgHom k A (g.comp f) = (mapDomainAlgHom k A g).comp (mapDomainAlgHom k A f) := by
  /-
    k : Type u_4
    A : Type u_5
    G₁ : Type u_6
    G₂ : Type u_7
    G₃ : Type u_8
    inst✝⁵ : CommSemiring k
    inst✝⁴ : Semiring A
    inst✝³ : Algebra k A
    inst✝² : Monoid G₁
    inst✝¹ : Monoid G₂
    inst✝ : Monoid G₃
    f : MonoidHom G₁ G₂
    g : MonoidHom G₂ G₃
    ⊢ Eq (MonoidAlgebra.mapDomainAlgHom k A (g.comp f)) ((MonoidAlgebra.mapDomainA …
  -/
  ext; simp [mapDomain_comp]
       /-
         🎉 no goals
       -/


/-- If `e : G ≃* H` is a multiplicative equivalence between two monoids, then
`MonoidAlgebra.domCongr e` is an algebra equivalence between their monoid algebras. -/
def domCongr (e : G ≃* H) : MonoidAlgebra A G ≃ₐ[k] MonoidAlgebra A H :=
  AlgEquiv.ofLinearEquiv
    (Finsupp.domLCongr e : (G →₀ A) ≃ₗ[k] (H →₀ A))
    ((equivMapDomain_eq_mapDomain _ _).trans <| mapDomain_one e)
    (fun f g => (equivMapDomain_eq_mapDomain _ _).trans <| (mapDomain_mul e f g).trans <|
        congr_arg₂ _ (equivMapDomain_eq_mapDomain _ _).symm (equivMapDomain_eq_mapDomain _ _).symm)


theorem domCongr_toAlgHom (e : G ≃* H) : (domCongr k A e).toAlgHom = mapDomainAlgHom k A e :=
  AlgHom.ext fun _ => equivMapDomain_eq_mapDomain _ _


@[simp] theorem domCongr_apply (e : G ≃* H) (f : MonoidAlgebra A G) (h : H) :
    domCongr k A e f h = f (e.symm h) :=
  rfl


@[simp] theorem domCongr_support (e : G ≃* H) (f : MonoidAlgebra A G) :
    (domCongr k A e f).support = f.support.map e :=
  rfl


@[simp] theorem domCongr_single (e : G ≃* H) (g : G) (a : A) :
    domCongr k A e (single g a) = single (e g) a :=
  Finsupp.equivMapDomain_single _ _ _


@[simp] theorem domCongr_refl : domCongr k A (MulEquiv.refl G) = AlgEquiv.refl :=
  AlgEquiv.ext fun _ => Finsupp.ext fun _ => rfl


@[simp] theorem domCongr_symm (e : G ≃* H) : (domCongr k A e).symm = domCongr k A e.symm := rfl


/-- When `V` is a `k[G]`-module, multiplication by a group element `g` is a `k`-linear map. -/
def GroupSMul.linearMap [Monoid G] [CommSemiring k] (V : Type u₃) [AddCommMonoid V] [Module k V]
    [Module (MonoidAlgebra k G) V] [IsScalarTower k (MonoidAlgebra k G) V] (g : G) : V →ₗ[k] V where
  toFun v := single g (1 : k) • v
  map_add' x y := smul_add (single g (1 : k)) x y
  map_smul' _c _x := smul_algebra_smul_comm _ _ _


@[simp]
theorem GroupSMul.linearMap_apply [Monoid G] [CommSemiring k] (V : Type u₃) [AddCommMonoid V]
    [Module k V] [Module (MonoidAlgebra k G) V] [IsScalarTower k (MonoidAlgebra k G) V] (g : G)
    (v : V) : (GroupSMul.linearMap k V g) v = single g (1 : k) • v :=
  rfl


/-- Build a `k[G]`-linear map from a `k`-linear map and evidence that it is `G`-equivariant. -/
def equivariantOfLinearOfComm
    (h : ∀ (g : G) (v : V), f (single g (1 : k) • v) = single g (1 : k) • f v) :
    V →ₗ[MonoidAlgebra k G] W where
  toFun := f
                      /-
                        k : Type u₁
                        G : Type u₂
                        H : Type u_1
                        R : Type u_2
                        inst✝⁹ : Monoid G
                        inst✝⁸ : CommSemiring k
                        V : Type u₃
                        W : Type u₄
                        inst✝⁷ : AddCommMonoid V
                        inst✝⁶ : Module k V
                        inst✝⁵ : Module (MonoidAlgebra k G) V
                        inst✝⁴ : IsScalarTower k (MonoidAlgebra k G) V
                        inst✝³ : AddCommMonoid W
                        inst✝² : Module k W
                        inst✝¹ : Module (MonoidAlgebra k G) W
                        inst✝ : IsScalarTower k (MonoidAlgebra k G) W
                        f : LinearMap (RingHom.id k) V W
                        h : ∀ (g : G) (v : V), Eq (f (HSMul.hSMul (MonoidAlgebra.single g 1) v)) (HSMu …
                        v v' : V
                        ⊢ Eq (f (HAdd.hAdd v v')) (HAdd.hAdd (f v) (f v'))
                      -/
  map_add' v v' := by simp
                      /-
                        🎉 no goals
                      -/
  map_smul' c v := by
    -- Porting note: Was `apply`.
    /-
      k : Type u₁
      G : Type u₂
      H : Type u_1
      R : Type u_2
      inst✝⁹ : Monoid G
      inst✝⁸ : CommSemiring k
      V : Type u₃
      W : Type u₄
      inst✝⁷ : AddCommMonoid V
      inst✝⁶ : Module k V
      inst✝⁵ : Module (MonoidAlgebra k G) V
      inst✝⁴ : IsScalarTower k (MonoidAlgebra k G) V
      inst✝³ : AddCommMonoid W
      inst✝² : Module k W
      inst✝¹ : Module (MonoidAlgebra k G) W
      inst✝ : IsScalarTower k (MonoidAlgebra k G) W
      f : LinearMap (RingHom.id k) V W
      h : ∀ (g : G) (v : V), Eq (f (HSMul.hSMul (MonoidAlgebra.single g 1) v)) (HSMu …
      c : MonoidAlgebra k G
      v : V
      ⊢ Eq ({ toFun := ⇑f, map_add' := ⋯ }.toFun (HSMul.hSMul c v)) (HSMul.hSMul ((R …
    -/
    refine Finsupp.induction c ?_ ?_
      /-
        case refine_1
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        inst✝⁹ : Monoid G
        inst✝⁸ : CommSemiring k
        V : Type u₃
        W : Type u₄
        inst✝⁷ : AddCommMonoid V
        inst✝⁶ : Module k V
        inst✝⁵ : Module (MonoidAlgebra k G) V
        inst✝⁴ : IsScalarTower k (MonoidAlgebra k G) V
        inst✝³ : AddCommMonoid W
        inst✝² : Module k W
        inst✝¹ : Module (MonoidAlgebra k G) W
        inst✝ : IsScalarTower k (MonoidAlgebra k G) W
        f : LinearMap (RingHom.id k) V W
        h : ∀ (g : G) (v : V), Eq (f (HSMul.hSMul (MonoidAlgebra.single g 1) v)) (HSMu …
        c : MonoidAlgebra k G
        v : V
        ⊢ Eq ({ toFun := ⇑f, map_add' := ⋯ }.toFun (HSMul.hSMul 0 v)) (HSMul.hSMul ((R …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        inst✝⁹ : Monoid G
        inst✝⁸ : CommSemiring k
        V : Type u₃
        W : Type u₄
        inst✝⁷ : AddCommMonoid V
        inst✝⁶ : Module k V
        inst✝⁵ : Module (MonoidAlgebra k G) V
        inst✝⁴ : IsScalarTower k (MonoidAlgebra k G) V
        inst✝³ : AddCommMonoid W
        inst✝² : Module k W
        inst✝¹ : Module (MonoidAlgebra k G) W
        inst✝ : IsScalarTower k (MonoidAlgebra k G) W
        f : LinearMap (RingHom.id k) V W
        h : ∀ (g : G) (v : V), Eq (f (HSMul.hSMul (MonoidAlgebra.single g 1) v)) (HSMu …
        c : MonoidAlgebra k G
        v : V
        ⊢ ∀ (a : G) (b : k) (f_1 : Finsupp G k), Not (Membership.mem f_1.support a) →  …
      -/
    · intro g r c' _nm _nz w
      /-
        case refine_2
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        inst✝⁹ : Monoid G
        inst✝⁸ : CommSemiring k
        V : Type u₃
        W : Type u₄
        inst✝⁷ : AddCommMonoid V
        inst✝⁶ : Module k V
        inst✝⁵ : Module (MonoidAlgebra k G) V
        inst✝⁴ : IsScalarTower k (MonoidAlgebra k G) V
        inst✝³ : AddCommMonoid W
        inst✝² : Module k W
        inst✝¹ : Module (MonoidAlgebra k G) W
        inst✝ : IsScalarTower k (MonoidAlgebra k G) W
        f : LinearMap (RingHom.id k) V W
        h : ∀ (g : G) (v : V), Eq (f (HSMul.hSMul (MonoidAlgebra.single g 1) v)) (HSMu …
        c : MonoidAlgebra k G
        v : V
        g : G
        r : k
        c' : Finsupp G k
        _nm : Not (Membership.mem c'.support g)
        _nz : Ne r 0
        w : Eq ({ toFun := ⇑f, map_add' := ⋯ }.toFun (HSMul.hSMul c' v)) (HSMul.hSMul  …
        ⊢ Eq ({ toFun := ⇑f, map_add' := ⋯ }.toFun (HSMul.hSMul (HAdd.hAdd (Finsupp.si …
      -/
      dsimp at *
      /-
        case refine_2
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        inst✝⁹ : Monoid G
        inst✝⁸ : CommSemiring k
        V : Type u₃
        W : Type u₄
        inst✝⁷ : AddCommMonoid V
        inst✝⁶ : Module k V
        inst✝⁵ : Module (MonoidAlgebra k G) V
        inst✝⁴ : IsScalarTower k (MonoidAlgebra k G) V
        inst✝³ : AddCommMonoid W
        inst✝² : Module k W
        inst✝¹ : Module (MonoidAlgebra k G) W
        inst✝ : IsScalarTower k (MonoidAlgebra k G) W
        f : LinearMap (RingHom.id k) V W
        h : ∀ (g : G) (v : V), Eq (f (HSMul.hSMul (MonoidAlgebra.single g 1) v)) (HSMu …
        c : MonoidAlgebra k G
        v : V
        g : G
        r : k
        c' : Finsupp G k
        _nm : Not (Membership.mem c'.support g)
        _nz : Not (Eq r 0)
        w : Eq (f (HSMul.hSMul c' v)) (HSMul.hSMul c' (f v))
        ⊢ Eq (f (HSMul.hSMul (HAdd.hAdd (Finsupp.single g r) c') v)) (HSMul.hSMul (HAd …
      -/
      simp only [add_smul, f.map_add, w, add_left_inj, single_eq_algebraMap_mul_of, ← smul_smul]
      erw [algebraMap_smul (MonoidAlgebra k G) r, algebraMap_smul (MonoidAlgebra k G) r, f.map_smul,
        h g v, of_apply]


@[simp]
theorem equivariantOfLinearOfComm_apply (v : V) : (equivariantOfLinearOfComm f h) v = f v :=
  rfl


/-- A non_unital `k`-algebra homomorphism from `k[G]` is uniquely defined by its
values on the functions `single a 1`. -/
theorem nonUnitalAlgHom_ext [DistribMulAction k A] {φ₁ φ₂ : k[G] →ₙₐ[k] A}
    (h : ∀ x, φ₁ (single x 1) = φ₂ (single x 1)) : φ₁ = φ₂ :=
  @MonoidAlgebra.nonUnitalAlgHom_ext k (Multiplicative G) _ _ _ _ _ φ₁ φ₂ h


/-- See note [partially-applied ext lemmas]. -/
@[ext high]
theorem nonUnitalAlgHom_ext' [DistribMulAction k A] {φ₁ φ₂ : k[G] →ₙₐ[k] A}
    (h : φ₁.toMulHom.comp (ofMagma k G) = φ₂.toMulHom.comp (ofMagma k G)) : φ₁ = φ₂ :=
  @MonoidAlgebra.nonUnitalAlgHom_ext' k (Multiplicative G) _ _ _ _ _ φ₁ φ₂ h


/-- The functor `G ↦ k[G]`, from the category of magmas to the category of
non-unital, non-associative algebras over `k` is adjoint to the forgetful functor in the other
direction. -/
@[simps apply_apply symm_apply]
def liftMagma [Module k A] [IsScalarTower k A A] [SMulCommClass k A A] :
    (Multiplicative G →ₙ* A) ≃ (k[G] →ₙₐ[k] A) :=
  { (MonoidAlgebra.liftMagma k : (Multiplicative G →ₙ* A) ≃ (_ →ₙₐ[k] A)) with
    toFun := fun f =>
      { (MonoidAlgebra.liftMagma k f : _) with
        toFun := fun a => sum a fun m t => t • f (Multiplicative.ofAdd m) }
    invFun := fun F => F.toMulHom.comp (ofMagma k G) }


/-- The instance `Algebra R k[G]` whenever we have `Algebra R k`.

In particular this provides the instance `Algebra k k[G]`.
-/
instance algebra [CommSemiring R] [Semiring k] [Algebra R k] [AddMonoid G] :
    Algebra R k[G] :=
  { singleZeroRingHom.comp (algebraMap R k) with
    smul_def' := fun r a => by
      /-
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : Semiring k
        inst✝¹ : Algebra R k
        inst✝ : AddMonoid G
        r : R
        a : AddMonoidAlgebra k G
        ⊢ Eq (HSMul.hSMul r a) (HMul.hMul (__src✝ r) a)
      -/
      ext
      -- Porting note: Newly required.
      /-
        case H
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : Semiring k
        inst✝¹ : Algebra R k
        inst✝ : AddMonoid G
        r : R
        a : AddMonoidAlgebra k G
        x✝ : G
        ⊢ Eq ((HSMul.hSMul r a) x✝) ((HMul.hMul (__src✝ r) a) x✝)
      -/
      rw [Finsupp.coe_smul]
      /-
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : Semiring k
        inst✝¹ : Algebra R k
        inst✝ : AddMonoid G
        r : R
        f : AddMonoidAlgebra k G
        ⊢ Eq (HMul.hMul (__src✝ r) f) (HMul.hMul f (__src✝ r))
      -/
      /-
        case H
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : Semiring k
        inst✝¹ : Algebra R k
        inst✝ : AddMonoid G
        r : R
        a : AddMonoidAlgebra k G
        x✝ : G
        ⊢ Eq (HSMul.hSMul r (⇑a) x✝) ((HMul.hMul (__src✝ r) a) x✝)
      -/
      /-
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : Semiring k
        inst✝¹ : Algebra R k
        inst✝ : AddMonoid G
        r : R
        f : AddMonoidAlgebra k G
        x✝ : G
        ⊢ Eq ((HMul.hMul (__src✝ r) f) x✝) ((HMul.hMul f (__src✝ r)) x✝)
      -/
      simp [single_zero_mul_apply, Algebra.smul_def, Pi.smul_apply]
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
    commutes' := fun r f => by
      refine Finsupp.ext fun _ => ?_
      simp [single_zero_mul_apply, mul_single_zero_apply, Algebra.commutes] }


/-- `Finsupp.single 0` as an `AlgHom` -/
@[simps! apply]
def singleZeroAlgHom [CommSemiring R] [Semiring k] [Algebra R k] [AddMonoid G] : k →ₐ[R] k[G] :=
  { singleZeroRingHom with
    commutes' := fun r => by
      /-
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : Semiring k
        inst✝¹ : Algebra R k
        inst✝ : AddMonoid G
        r : R
        ⊢ Eq ((↑↑__src✝).toFun ((algebraMap R k) r)) ((algebraMap R (AddMonoidAlgebra  …
      -/
      ext
      /-
        case H
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : Semiring k
        inst✝¹ : Algebra R k
        inst✝ : AddMonoid G
        r : R
        x✝ : G
        ⊢ Eq (((↑↑__src✝).toFun ((algebraMap R k) r)) x✝) (((algebraMap R (AddMonoidAl …
      -/
      simp
      /-
        case H
        k : Type u₁
        G : Type u₂
        H : Type u_1
        R : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : Semiring k
        inst✝¹ : Algebra R k
        inst✝ : AddMonoid G
        r : R
        x✝ : G
        ⊢ Eq ((Finsupp.single 0 ((algebraMap R k) r)) x✝) (((algebraMap R (AddMonoidAl …
      -/
      rfl }
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_algebraMap [CommSemiring R] [Semiring k] [Algebra R k] [AddMonoid G] :
    (algebraMap R k[G] : R → k[G]) = single 0 ∘ algebraMap R k :=
  rfl


/-- `liftNCRingHom` as an `AlgHom`, for when `f` is an `AlgHom` -/
def liftNCAlgHom (f : A →ₐ[k] B) (g : Multiplicative G →* B) (h_comm : ∀ x y, Commute (f x) (g y)) :
    A[G] →ₐ[k] B :=
  { liftNCRingHom (f : A →+* B) g h_comm with
                    /-
                      k : Type u₁
                      G : Type u₂
                      H : Type u_1
                      R : Type u_2
                      inst✝⁵ : CommSemiring k
                      inst✝⁴ : AddMonoid G
                      A : Type u₃
                      inst✝³ : Semiring A
                      inst✝² : Algebra k A
                      B : Type u_3
                      inst✝¹ : Semiring B
                      inst✝ : Algebra k B
                      f : AlgHom k A B
                      g : MonoidHom (Multiplicative G) B
                      h_comm : ∀ (x : A) (y : Multiplicative G), Commute (f x) (g y)
                      ⊢ ∀ (r : k), Eq ((↑↑__src✝).toFun ((algebraMap k (AddMonoidAlgebra A G)) r)) ( …
                    -/
    commutes' := by simp [liftNCRingHom] }
                    /-
                      🎉 no goals
                    -/


/-- A `k`-algebra homomorphism from `k[G]` is uniquely defined by its
values on the functions `single a 1`. -/
theorem algHom_ext ⦃φ₁ φ₂ : k[G] →ₐ[k] A⦄
    (h : ∀ x, φ₁ (single x 1) = φ₂ (single x 1)) : φ₁ = φ₂ :=
  @MonoidAlgebra.algHom_ext k (Multiplicative G) _ _ _ _ _ _ _ h


/-- See note [partially-applied ext lemmas]. -/
@[ext high]
theorem algHom_ext' ⦃φ₁ φ₂ : k[G] →ₐ[k] A⦄
    (h : (φ₁ : k[G] →* A).comp (of k G) = (φ₂ : k[G] →* A).comp (of k G)) :
    φ₁ = φ₂ :=
  algHom_ext <| DFunLike.congr_fun h


/-- Any monoid homomorphism `G →* A` can be lifted to an algebra homomorphism
`k[G] →ₐ[k] A`. -/
def lift : (Multiplicative G →* A) ≃ (k[G] →ₐ[k] A) :=
  { @MonoidAlgebra.lift k (Multiplicative G) _ _ A _ _ with
    invFun := fun f => (f : k[G] →* A).comp (of k G)
    toFun := fun F =>
      { @MonoidAlgebra.lift k (Multiplicative G) _ _ A _ _ F with
        toFun := liftNCAlgHom (Algebra.ofId k A) F fun _ _ => Algebra.commutes _ _ } }


theorem lift_apply' (F : Multiplicative G →* A) (f : MonoidAlgebra k G) :
    lift k G A F f = f.sum fun a b => algebraMap k A b * F (Multiplicative.ofAdd a) :=
  rfl


theorem lift_apply (F : Multiplicative G →* A) (f : MonoidAlgebra k G) :
    lift k G A F f = f.sum fun a b => b • F (Multiplicative.ofAdd a) := by
  /-
    k : Type u₁
    G : Type u₂
    inst✝³ : CommSemiring k
    inst✝² : AddMonoid G
    A : Type u₃
    inst✝¹ : Semiring A
    inst✝ : Algebra k A
    F : MonoidHom (Multiplicative G) A
    f : MonoidAlgebra k G
    ⊢ Eq (((AddMonoidAlgebra.lift k G A) F) f) (Finsupp.sum f fun a b => HSMul.hSM …
  -/
  simp only [lift_apply', Algebra.smul_def]
  /-
    🎉 no goals
  -/


theorem lift_def (F : Multiplicative G →* A) :
    ⇑(lift k G A F) = liftNC ((algebraMap k A : k →+* A) : k →+ A) F :=
  rfl


@[simp]
theorem lift_symm_apply (F : k[G] →ₐ[k] A) (x : Multiplicative G) :
    (lift k G A).symm F x = F (single x.toAdd 1) :=
  rfl


theorem lift_of (F : Multiplicative G →* A) (x : Multiplicative G) :
    lift k G A F (of k G x) = F x := MonoidAlgebra.lift_of F x


@[simp]
theorem lift_single (F : Multiplicative G →* A) (a b) :
    lift k G A F (single a b) = b • F (Multiplicative.ofAdd a) :=
  MonoidAlgebra.lift_single F (.ofAdd a) b


lemma lift_of' (F : Multiplicative G →* A) (x : G) :
    lift k G A F (of' k G x) = F (Multiplicative.ofAdd x) :=
  lift_of F x


theorem lift_unique' (F : k[G] →ₐ[k] A) :
    F = lift k G A ((F : k[G] →* A).comp (of k G)) :=
  ((lift k G A).apply_symm_apply F).symm


/-- Decomposition of a `k`-algebra homomorphism from `MonoidAlgebra k G` by
its values on `F (single a 1)`. -/
theorem lift_unique (F : k[G] →ₐ[k] A) (f : MonoidAlgebra k G) :
    F f = f.sum fun a b => b • F (single a 1) := by
  conv_lhs =>
    rw [lift_unique' F]
    simp [lift_apply]


theorem algHom_ext_iff {φ₁ φ₂ : k[G] →ₐ[k] A} :
    (∀ x, φ₁ (Finsupp.single x 1) = φ₂ (Finsupp.single x 1)) ↔ φ₁ = φ₂ :=
                             /-
                               k : Type u₁
                               G : Type u₂
                               inst✝³ : CommSemiring k
                               inst✝² : AddMonoid G
                               A : Type u₃
                               inst✝¹ : Semiring A
                               inst✝ : Algebra k A
                               φ₁ φ₂ : AlgHom k (AddMonoidAlgebra k G) A
                               ⊢ Eq φ₁ φ₂ → ∀ (x : G), Eq (φ₁ (Finsupp.single x 1)) (φ₂ (Finsupp.single x 1))
                             -/
  ⟨fun h => algHom_ext h, by rintro rfl _; rfl⟩
                                           /-
                                             🎉 no goals
                                           -/


theorem mapDomain_algebraMap (A : Type*) {H F : Type*} [CommSemiring k] [Semiring A] [Algebra k A]
    [AddMonoid G] [AddMonoid H] [FunLike F G H] [AddMonoidHomClass F G H]
    (f : F) (r : k) :
    mapDomain f (algebraMap k A[G] r) = algebraMap k A[H] r := by
  /-
    k : Type u₁
    G : Type u₂
    A : Type u_3
    H : Type u_4
    F : Type u_5
    inst✝⁶ : CommSemiring k
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra k A
    inst✝³ : AddMonoid G
    inst✝² : AddMonoid H
    inst✝¹ : FunLike F G H
    inst✝ : AddMonoidHomClass F G H
    f : F
    r : k
    ⊢ Eq (AddMonoidAlgebra.mapDomain (⇑f) ((algebraMap k (AddMonoidAlgebra A G)) r …
  -/
  simp only [Function.comp_apply, mapDomain_single, AddMonoidAlgebra.coe_algebraMap, map_zero]
  /-
    🎉 no goals
  -/


/-- If `f : G → H` is a homomorphism between two additive magmas, then `Finsupp.mapDomain f` is a
non-unital algebra homomorphism between their additive magma algebras. -/
@[simps apply]
def mapDomainNonUnitalAlgHom (k A : Type*) [CommSemiring k] [Semiring A] [Algebra k A]
    {G H F : Type*} [Add G] [Add H] [FunLike F G H] [AddHomClass F G H] (f : F) :
    A[G] →ₙₐ[k] A[H] :=
  { (Finsupp.mapDomain.addMonoidHom f : MonoidAlgebra A G →+ MonoidAlgebra A H) with
    map_mul' := fun x y => mapDomain_mul f x y
    map_smul' := fun r x => mapDomain_smul r x }


/-- If `f : G → H` is an additive homomorphism between two additive monoids, then
`Finsupp.mapDomain f` is an algebra homomorphism between their add monoid algebras. -/
@[simps!]
def mapDomainAlgHom (k A : Type*) [CommSemiring k] [Semiring A] [Algebra k A] [AddMonoid G]
    {H F : Type*} [AddMonoid H] [FunLike F G H] [AddMonoidHomClass F G H] (f : F) :
    A[G] →ₐ[k] A[H] :=
  { mapDomainRingHom A f with commutes' := mapDomain_algebraMap A f }


@[simp]
lemma mapDomainAlgHom_id (k A) [CommSemiring k] [Semiring A] [Algebra k A] [AddMonoid G] :
    mapDomainAlgHom k A (AddMonoidHom.id G) = AlgHom.id k (AddMonoidAlgebra A G) := by
  /-
    G : Type u₂
    k : Type u_3
    A : Type u_4
    inst✝³ : CommSemiring k
    inst✝² : Semiring A
    inst✝¹ : Algebra k A
    inst✝ : AddMonoid G
    ⊢ Eq (AddMonoidAlgebra.mapDomainAlgHom k A (AddMonoidHom.id G)) (AlgHom.id k ( …
  -/
  ext; simp [AddMonoidHom.id, ← Function.id_def]
       /-
         🎉 no goals
       -/


@[simp]
lemma mapDomainAlgHom_comp (k A) {G₁ G₂ G₃} [CommSemiring k] [Semiring A] [Algebra k A]
    [AddMonoid G₁] [AddMonoid G₂] [AddMonoid G₃] (f : G₁ →+ G₂) (g : G₂ →+ G₃) :
    mapDomainAlgHom k A (g.comp f) = (mapDomainAlgHom k A g).comp (mapDomainAlgHom k A f) := by
  /-
    k : Type u_3
    A : Type u_4
    G₁ : Type u_5
    G₂ : Type u_6
    G₃ : Type u_7
    inst✝⁵ : CommSemiring k
    inst✝⁴ : Semiring A
    inst✝³ : Algebra k A
    inst✝² : AddMonoid G₁
    inst✝¹ : AddMonoid G₂
    inst✝ : AddMonoid G₃
    f : AddMonoidHom G₁ G₂
    g : AddMonoidHom G₂ G₃
    ⊢ Eq (AddMonoidAlgebra.mapDomainAlgHom k A (g.comp f)) ((AddMonoidAlgebra.mapD …
  -/
  ext; simp [mapDomain_comp]
       /-
         🎉 no goals
       -/


/-- If `e : G ≃* H` is a multiplicative equivalence between two monoids, then
`AddMonoidAlgebra.domCongr e` is an algebra equivalence between their monoid algebras. -/
def domCongr (e : G ≃+ H) : A[G] ≃ₐ[k] A[H] :=
  AlgEquiv.ofLinearEquiv
    (Finsupp.domLCongr e : (G →₀ A) ≃ₗ[k] (H →₀ A))
    ((equivMapDomain_eq_mapDomain _ _).trans <| mapDomain_one e)
    (fun f g => (equivMapDomain_eq_mapDomain _ _).trans <| (mapDomain_mul e f g).trans <|
        congr_arg₂ _ (equivMapDomain_eq_mapDomain _ _).symm (equivMapDomain_eq_mapDomain _ _).symm)


theorem domCongr_toAlgHom (e : G ≃+ H) : (domCongr k A e).toAlgHom = mapDomainAlgHom k A e :=
  AlgHom.ext fun _ => equivMapDomain_eq_mapDomain _ _


@[simp] theorem domCongr_apply (e : G ≃+ H) (f : MonoidAlgebra A G) (h : H) :
    domCongr k A e f h = f (e.symm h) :=
  rfl


@[simp] theorem domCongr_support (e : G ≃+ H) (f : MonoidAlgebra A G) :
    (domCongr k A e f).support = f.support.map e :=
  rfl


@[simp] theorem domCongr_single (e : G ≃+ H) (g : G) (a : A) :
    domCongr k A e (single g a) = single (e g) a :=
  Finsupp.equivMapDomain_single _ _ _


@[simp] theorem domCongr_refl : domCongr k A (AddEquiv.refl G) = AlgEquiv.refl :=
  AlgEquiv.ext fun _ => Finsupp.ext fun _ => rfl


@[simp] theorem domCongr_symm (e : G ≃+ H) : (domCongr k A e).symm = domCongr k A e.symm := rfl


/-- The algebra equivalence between `AddMonoidAlgebra` and `MonoidAlgebra` in terms of
`Multiplicative`. -/
def AddMonoidAlgebra.toMultiplicativeAlgEquiv [Semiring k] [Algebra R k] [AddMonoid G] :
    AddMonoidAlgebra k G ≃ₐ[R] MonoidAlgebra k (Multiplicative G) :=
  { AddMonoidAlgebra.toMultiplicative k G with
                             /-
                               k : Type u₁
                               G : Type u₂
                               H : Type u_1
                               R : Type u_2
                               inst✝³ : CommSemiring R
                               inst✝² : Semiring k
                               inst✝¹ : Algebra R k
                               inst✝ : AddMonoid G
                               r : R
                               ⊢ Eq (__src✝.toFun ((algebraMap R (AddMonoidAlgebra k G)) r)) ((algebraMap R ( …
                             -/
    commutes' := fun r => by simp [AddMonoidAlgebra.toMultiplicative] }
                             /-
                               🎉 no goals
                             -/


/-- The algebra equivalence between `MonoidAlgebra` and `AddMonoidAlgebra` in terms of
`Additive`. -/
def MonoidAlgebra.toAdditiveAlgEquiv [Semiring k] [Algebra R k] [Monoid G] :
    MonoidAlgebra k G ≃ₐ[R] AddMonoidAlgebra k (Additive G) :=
                                                               /-
                                                                 k : Type u₁
                                                                 G : Type u₂
                                                                 H : Type u_1
                                                                 R : Type u_2
                                                                 inst✝³ : CommSemiring R
                                                                 inst✝² : Semiring k
                                                                 inst✝¹ : Algebra R k
                                                                 inst✝ : Monoid G
                                                                 r : R
                                                                 ⊢ Eq (__src✝.toFun ((algebraMap R (MonoidAlgebra k G)) r)) ((algebraMap R (Add …
                                                               -/
  { MonoidAlgebra.toAdditive k G with commutes' := fun r => by simp [MonoidAlgebra.toAdditive] }
                                                               /-
                                                                 🎉 no goals
                                                               -/

