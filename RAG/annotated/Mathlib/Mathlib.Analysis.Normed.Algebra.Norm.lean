/-- An algebra norm on an `R`-algebra `S` is a ring norm on `S` compatible with the
action of `R`. -/
structure AlgebraNorm (R : Type*) [SeminormedCommRing R] (S : Type*) [Ring S] [Algebra R S] extends
  RingNorm S, Seminorm R S


instance (K : Type*) [NormedField K] : Inhabited (AlgebraNorm K K) :=
  ⟨{  toFun     := norm
      map_zero' := norm_zero
      add_le'   := norm_add_le
      neg'      := norm_neg
      smul'     := norm_mul
      mul_le'   := norm_mul_le
      eq_zero_of_map_eq_zero' := fun _ => norm_eq_zero.mp }⟩


/-- `AlgebraNormClass F R S` states that `F` is a type of `R`-algebra norms on the ring `S`.
You should extend this class when you extend `AlgebraNorm`. -/
class AlgebraNormClass (F : Type*) (R : outParam <| Type*) [SeminormedCommRing R]
    (S : outParam <| Type*) [Ring S] [Algebra R S] [FunLike F S ℝ] extends RingNormClass F S ℝ,
    SeminormClass F R S : Prop


/-- The ring seminorm underlying an algebra norm. -/
def toRingSeminorm' (f : AlgebraNorm R S) : RingSeminorm S :=
  f.toRingNorm.toRingSeminorm


instance : FunLike (AlgebraNorm R S) S ℝ where
  coe f := f.toFun
  coe_injective' f f' h := by
    /-
      R : Type u_1
      inst✝² : SeminormedCommRing R
      S : Type u_2
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      f✝ f f' : AlgebraNorm R S
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) f')
      ⊢ Eq f f'
    -/
    simp only [AddGroupSeminorm.toFun_eq_coe, RingSeminorm.toFun_eq_coe] at h
    /-
      R : Type u_1
      inst✝² : SeminormedCommRing R
      S : Type u_2
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      f✝ f f' : AlgebraNorm R S
      h : Eq ⇑f.toRingSeminorm ⇑f'.toRingSeminorm
      ⊢ Eq f f'
    -/
    cases f; cases f'; congr
    /-
      case mk.mk.e_toRingNorm
      R : Type u_1
      inst✝² : SeminormedCommRing R
      S : Type u_2
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      f : AlgebraNorm R S
      toRingNorm✝¹ : RingNorm S
      smul'✝¹ : ∀ (a : R) (x : S), Eq (toRingNorm✝¹.toFun (HSMul.hSMul a x)) (HMul.h …
      toRingNorm✝ : RingNorm S
      smul'✝ : ∀ (a : R) (x : S), Eq (toRingNorm✝.toFun (HSMul.hSMul a x)) (HMul.hMu …
      h : Eq ⇑{ toRingNorm := toRingNorm✝¹, smul' := smul'✝¹ }.toRingSeminorm ⇑{ toR …
      ⊢ Eq toRingNorm✝¹ toRingNorm✝
    -/
    simp only at h
    /-
      case mk.mk.e_toRingNorm
      R : Type u_1
      inst✝² : SeminormedCommRing R
      S : Type u_2
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      f : AlgebraNorm R S
      toRingNorm✝¹ : RingNorm S
      smul'✝¹ : ∀ (a : R) (x : S), Eq (toRingNorm✝¹.toFun (HSMul.hSMul a x)) (HMul.h …
      toRingNorm✝ : RingNorm S
      smul'✝ : ∀ (a : R) (x : S), Eq (toRingNorm✝.toFun (HSMul.hSMul a x)) (HMul.hMu …
      h : Eq ⇑toRingNorm✝¹.toRingSeminorm ⇑toRingNorm✝.toRingSeminorm
      ⊢ Eq toRingNorm✝¹ toRingNorm✝
    -/
    ext s
    /-
      case mk.mk.e_toRingNorm.a
      R : Type u_1
      inst✝² : SeminormedCommRing R
      S : Type u_2
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      f : AlgebraNorm R S
      toRingNorm✝¹ : RingNorm S
      smul'✝¹ : ∀ (a : R) (x : S), Eq (toRingNorm✝¹.toFun (HSMul.hSMul a x)) (HMul.h …
      toRingNorm✝ : RingNorm S
      smul'✝ : ∀ (a : R) (x : S), Eq (toRingNorm✝.toFun (HSMul.hSMul a x)) (HMul.hMu …
      h : Eq ⇑toRingNorm✝¹.toRingSeminorm ⇑toRingNorm✝.toRingSeminorm
      s : S
      ⊢ Eq (toRingNorm✝¹ s) (toRingNorm✝ s)
    -/
    erw [h]
    /-
      case mk.mk.e_toRingNorm.a
      R : Type u_1
      inst✝² : SeminormedCommRing R
      S : Type u_2
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      f : AlgebraNorm R S
      toRingNorm✝¹ : RingNorm S
      smul'✝¹ : ∀ (a : R) (x : S), Eq (toRingNorm✝¹.toFun (HSMul.hSMul a x)) (HMul.h …
      toRingNorm✝ : RingNorm S
      smul'✝ : ∀ (a : R) (x : S), Eq (toRingNorm✝.toFun (HSMul.hSMul a x)) (HMul.hMu …
      h : Eq ⇑toRingNorm✝¹.toRingSeminorm ⇑toRingNorm✝.toRingSeminorm
      s : S
      ⊢ Eq (toRingNorm✝.toRingSeminorm s) (toRingNorm✝ s)
    -/
    rfl
    /-
      🎉 no goals
    -/


instance algebraNormClass : AlgebraNormClass (AlgebraNorm R S) R S where
  map_zero f        := f.map_zero'
  map_add_le_add f  := f.add_le'
  map_mul_le_mul f  := f.mul_le'
  map_neg_eq_map f  := f.neg'
  eq_zero_of_map_eq_zero f := f.eq_zero_of_map_eq_zero' _
  map_smul_eq_mul f := f.smul'


theorem toFun_eq_coe (p : AlgebraNorm R S) : p.toFun = p := rfl


@[ext]
theorem ext {p q : AlgebraNorm R S} : (∀ x, p x = q x) → p = q :=
  DFunLike.ext p q


/-- An `R`-algebra norm such that `f 1 = 1` extends the norm on `R`. -/
theorem extends_norm' (hf1 : f 1 = 1) (a : R) : f (a • (1 : S)) = ‖a‖ := by
  /-
    R : Type u_1
    inst✝² : SeminormedCommRing R
    S : Type u_2
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    f : AlgebraNorm R S
    hf1 : Eq (f 1) 1
    a : R
    ⊢ Eq (f (HSMul.hSMul a 1)) (Norm.norm a)
  -/
  rw [← mul_one ‖a‖, ← hf1]; exact f.smul' _ _
                             /-
                               🎉 no goals
                             -/


/-- An `R`-algebra norm such that `f 1 = 1` extends the norm on `R`. -/
theorem extends_norm (hf1 : f 1 = 1) (a : R) : f (algebraMap R S a) = ‖a‖ := by
  /-
    R : Type u_1
    inst✝² : SeminormedCommRing R
    S : Type u_2
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    f : AlgebraNorm R S
    hf1 : Eq (f 1) 1
    a : R
    ⊢ Eq (f ((algebraMap R S) a)) (Norm.norm a)
  -/
  rw [Algebra.algebraMap_eq_smul_one]; exact extends_norm' hf1 _
                                       /-
                                         🎉 no goals
                                       -/


/-- The restriction of an algebra norm to a subalgebra. -/
def restriction (A : Subalgebra R S) (f : AlgebraNorm R S) : AlgebraNorm R A where
  toFun x     := f x.val
  map_zero'   := map_zero f
  add_le' x y := map_add_le_add _ _ _
  neg' x      := map_neg_eq_map _ _
  mul_le' x y := map_mul_le_mul _ _ _
  eq_zero_of_map_eq_zero' x hx := by
    /-
      R : Type u_1
      inst✝² : SeminormedCommRing R
      S : Type u_2
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      f✝ : AlgebraNorm R S
      A : Subalgebra R S
      f : AlgebraNorm R S
      x : Subtype fun x => Membership.mem A x
      hx : Eq ({ toFun := fun x => f ↑x, map_zero' := ⋯, add_le' := ⋯, neg' := ⋯, mu …
      ⊢ Eq x 0
    -/
    rw [← ZeroMemClass.coe_eq_zero]; exact eq_zero_of_map_eq_zero f hx
                                     /-
                                       🎉 no goals
                                     -/
  smul' r x := map_smul_eq_mul _ _ _


/-- The restriction of an algebra norm in a scalar tower. -/
def isScalarTower_restriction {A : Type*} [CommRing A] [Algebra R A] [Algebra A S]
    [IsScalarTower R A S] (hinj : Function.Injective (algebraMap A S)) (f : AlgebraNorm R S) :
    AlgebraNorm R A where
  toFun x     := f (algebraMap A S x)
                    /-
                      R : Type u_1
                      inst✝⁶ : SeminormedCommRing R
                      S : Type u_2
                      inst✝⁵ : Ring S
                      inst✝⁴ : Algebra R S
                      f✝ : AlgebraNorm R S
                      A : Type u_3
                      inst✝³ : CommRing A
                      inst✝² : Algebra R A
                      inst✝¹ : Algebra A S
                      inst✝ : IsScalarTower R A S
                      hinj : Function.Injective ⇑(algebraMap A S)
                      f : AlgebraNorm R S
                      ⊢ Eq ((fun x => f ((algebraMap A S) x)) 0) 0
                    -/
  map_zero'   := by simp only [map_zero]
                    /-
                      🎉 no goals
                    -/
                    /-
                      R : Type u_1
                      inst✝⁶ : SeminormedCommRing R
                      S : Type u_2
                      inst✝⁵ : Ring S
                      inst✝⁴ : Algebra R S
                      f✝ : AlgebraNorm R S
                      A : Type u_3
                      inst✝³ : CommRing A
                      inst✝² : Algebra R A
                      inst✝¹ : Algebra A S
                      inst✝ : IsScalarTower R A S
                      hinj : Function.Injective ⇑(algebraMap A S)
                      f : AlgebraNorm R S
                      x y : A
                      ⊢ LE.le ((fun x => f ((algebraMap A S) x)) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x …
                    -/
  add_le' x y := by simp only [map_add, map_add_le_add]
                    /-
                      🎉 no goals
                    -/
                    /-
                      R : Type u_1
                      inst✝⁶ : SeminormedCommRing R
                      S : Type u_2
                      inst✝⁵ : Ring S
                      inst✝⁴ : Algebra R S
                      f✝ : AlgebraNorm R S
                      A : Type u_3
                      inst✝³ : CommRing A
                      inst✝² : Algebra R A
                      inst✝¹ : Algebra A S
                      inst✝ : IsScalarTower R A S
                      hinj : Function.Injective ⇑(algebraMap A S)
                      f : AlgebraNorm R S
                      x : A
                      ⊢ Eq ((fun x => f ((algebraMap A S) x)) (Neg.neg x)) ((fun x => f ((algebraMap …
                    -/
  neg' x      := by simp only [map_neg, map_neg_eq_map]
                    /-
                      🎉 no goals
                    -/
                    /-
                      R : Type u_1
                      inst✝⁶ : SeminormedCommRing R
                      S : Type u_2
                      inst✝⁵ : Ring S
                      inst✝⁴ : Algebra R S
                      f✝ : AlgebraNorm R S
                      A : Type u_3
                      inst✝³ : CommRing A
                      inst✝² : Algebra R A
                      inst✝¹ : Algebra A S
                      inst✝ : IsScalarTower R A S
                      hinj : Function.Injective ⇑(algebraMap A S)
                      f : AlgebraNorm R S
                      x y : A
                      ⊢ LE.le ({ toFun := fun x => f ((algebraMap A S) x), map_zero' := ⋯, add_le' : …
                    -/
  mul_le' x y := by simp only [map_mul, map_mul_le_mul]
                    /-
                      🎉 no goals
                    -/
  eq_zero_of_map_eq_zero' x hx := by
    /-
      R : Type u_1
      inst✝⁶ : SeminormedCommRing R
      S : Type u_2
      inst✝⁵ : Ring S
      inst✝⁴ : Algebra R S
      f✝ : AlgebraNorm R S
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Algebra A S
      inst✝ : IsScalarTower R A S
      hinj : Function.Injective ⇑(algebraMap A S)
      f : AlgebraNorm R S
      x : A
      hx : Eq ({ toFun := fun x => f ((algebraMap A S) x), map_zero' := ⋯, add_le' : …
      ⊢ Eq x 0
    -/
    rw [← map_eq_zero_iff (algebraMap A S) hinj]
    /-
      R : Type u_1
      inst✝⁶ : SeminormedCommRing R
      S : Type u_2
      inst✝⁵ : Ring S
      inst✝⁴ : Algebra R S
      f✝ : AlgebraNorm R S
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Algebra A S
      inst✝ : IsScalarTower R A S
      hinj : Function.Injective ⇑(algebraMap A S)
      f : AlgebraNorm R S
      x : A
      hx : Eq ({ toFun := fun x => f ((algebraMap A S) x), map_zero' := ⋯, add_le' : …
      ⊢ Eq ((algebraMap A S) x) 0
    -/
    exact eq_zero_of_map_eq_zero f hx
    /-
      🎉 no goals
    -/
  smul' r x := by
    /-
      R : Type u_1
      inst✝⁶ : SeminormedCommRing R
      S : Type u_2
      inst✝⁵ : Ring S
      inst✝⁴ : Algebra R S
      f✝ : AlgebraNorm R S
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Algebra A S
      inst✝ : IsScalarTower R A S
      hinj : Function.Injective ⇑(algebraMap A S)
      f : AlgebraNorm R S
      r : R
      x : A
      ⊢ Eq ({ toFun := fun x => f ((algebraMap A S) x), map_zero' := ⋯, add_le' := ⋯ …
    -/
    simp only [Algebra.smul_def, map_mul, ← IsScalarTower.algebraMap_apply]
    /-
      R : Type u_1
      inst✝⁶ : SeminormedCommRing R
      S : Type u_2
      inst✝⁵ : Ring S
      inst✝⁴ : Algebra R S
      f✝ : AlgebraNorm R S
      A : Type u_3
      inst✝³ : CommRing A
      inst✝² : Algebra R A
      inst✝¹ : Algebra A S
      inst✝ : IsScalarTower R A S
      hinj : Function.Injective ⇑(algebraMap A S)
      f : AlgebraNorm R S
      r : R
      x : A
      ⊢ Eq (f (HMul.hMul ((algebraMap R S) r) ((algebraMap A S) x))) (HMul.hMul (Nor …
    -/
    simp only [← smul_eq_mul, algebraMap_smul, map_smul_eq_mul]
    /-
      🎉 no goals
    -/


/-- A multiplicative algebra norm on an `R`-algebra norm `S` is a multiplicative ring norm on `S`
  compatible with the action of `R`. -/
structure MulAlgebraNorm (R : Type*) [SeminormedCommRing R] (S : Type*) [Ring S] [Algebra R S]
  extends MulRingNorm S, Seminorm R S


instance (K : Type*) [NormedField K] : Inhabited (MulAlgebraNorm K K) :=
  ⟨{  toFun     := norm
      map_zero' := norm_zero
      add_le'   := norm_add_le
      neg'      := norm_neg
      smul'     := norm_mul
      map_one'  := norm_one
      map_mul'  := norm_mul
      eq_zero_of_map_eq_zero' := fun _ => norm_eq_zero.mp }⟩


/-- `MulAlgebraNormClass F R S` states that `F` is a type of multiplicative `R`-algebra norms on
the ring `S`. You should extend this class when you extend `MulAlgebraNorm`. -/
class MulAlgebraNormClass (F : Type*) (R : outParam <| Type*) [SeminormedCommRing R]
    (S : outParam <| Type*) [Ring S] [Algebra R S] [FunLike F S ℝ] extends MulRingNormClass F S ℝ,
    SeminormClass F R S : Prop


instance : FunLike (MulAlgebraNorm R S) S ℝ where
  coe f := f.toFun
  coe_injective' f f' h:= by
    /-
      R : outParam (Type u_1)
      S : outParam (Type u_2)
      inst✝² : SeminormedCommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      f✝ : AlgebraNorm R S
      f f' : MulAlgebraNorm R S
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) f')
      ⊢ Eq f f'
    -/
    simp only [AddGroupSeminorm.toFun_eq_coe, MulRingSeminorm.toFun_eq_coe, DFunLike.coe_fn_eq] at h
    /-
      R : outParam (Type u_1)
      S : outParam (Type u_2)
      inst✝² : SeminormedCommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      f✝ : AlgebraNorm R S
      f f' : MulAlgebraNorm R S
      h : Eq f.toMulRingSeminorm f'.toMulRingSeminorm
      ⊢ Eq f f'
    -/
    obtain ⟨⟨_, _⟩, _⟩ := f; obtain ⟨⟨_, _⟩, _⟩ := f'; congr
                                                       /-
                                                         🎉 no goals
                                                       -/


instance mulAlgebraNormClass : MulAlgebraNormClass (MulAlgebraNorm R S) R S where
  map_zero f        := f.map_zero'
  map_add_le_add f  := f.add_le'
  map_one f         := f.map_one'
  map_mul f         := f.map_mul'
  map_neg_eq_map f  := f.neg'
  eq_zero_of_map_eq_zero f := f.eq_zero_of_map_eq_zero' _
  map_smul_eq_mul f := f.smul'


theorem toFun_eq_coe (p : MulAlgebraNorm R S) : p.toFun = p := rfl


@[ext]
theorem ext {p q : MulAlgebraNorm R S} : (∀ x, p x = q x) → p = q :=
  DFunLike.ext p q


/-- A multiplicative `R`-algebra norm extends the norm on `R`. -/
theorem extends_norm' (f : MulAlgebraNorm R S) (a : R) : f (a • (1 : S)) = ‖a‖ := by
  /-
    R : outParam (Type u_1)
    S : outParam (Type u_2)
    inst✝² : SeminormedCommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    f : MulAlgebraNorm R S
    a : R
    ⊢ Eq (f (HSMul.hSMul a 1)) (Norm.norm a)
  -/
  rw [← mul_one ‖a‖, ← f.map_one', ← f.smul', toFun_eq_coe]
  /-
    🎉 no goals
  -/


/-- A multiplicative `R`-algebra norm extends the norm on `R`. -/
theorem extends_norm (f : MulAlgebraNorm R S) (a : R) : f (algebraMap R S a) = ‖a‖ := by
  /-
    R : outParam (Type u_1)
    S : outParam (Type u_2)
    inst✝² : SeminormedCommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    f : MulAlgebraNorm R S
    a : R
    ⊢ Eq (f ((algebraMap R S) a)) (Norm.norm a)
  -/
  rw [Algebra.algebraMap_eq_smul_one]; exact extends_norm' _ _
                                       /-
                                         🎉 no goals
                                       -/


/-- The ring norm underlying a multiplicative ring norm. -/
def toRingNorm (f : MulRingNorm R) : RingNorm R where
  toFun       := f
  map_zero'   := f.map_zero'
  add_le'     := f.add_le'
  neg'        := f.neg'
  mul_le' x y := le_of_eq (f.map_mul' x y)
  eq_zero_of_map_eq_zero' := f.eq_zero_of_map_eq_zero'


/-- A multiplicative ring norm is power-multiplicative. -/
theorem isPowMul {A : Type*} [Ring A] (f : MulRingNorm A) : IsPowMul f := fun x n hn => by
  /-
    A : Type u_2
    inst✝ : Ring A
    f : MulRingNorm A
    x : A
    n : Nat
    hn : LE.le 1 n
    ⊢ Eq (f (HPow.hPow x n)) (HPow.hPow (f x) n)
  -/
  cases n
    /-
      case zero
      A : Type u_2
      inst✝ : Ring A
      f : MulRingNorm A
      x : A
      hn : LE.le 1 0
      ⊢ Eq (f (HPow.hPow x 0)) (HPow.hPow (f x) 0)
    -/
  · omega
    /-
      🎉 no goals
    -/
    /-
      case succ
      A : Type u_2
      inst✝ : Ring A
      f : MulRingNorm A
      x : A
      n✝ : Nat
      hn : LE.le 1 (HAdd.hAdd n✝ 1)
      ⊢ Eq (f (HPow.hPow x (HAdd.hAdd n✝ 1))) (HPow.hPow (f x) (HAdd.hAdd n✝ 1))
    -/
  · rw [map_pow]
    /-
      🎉 no goals
    -/


