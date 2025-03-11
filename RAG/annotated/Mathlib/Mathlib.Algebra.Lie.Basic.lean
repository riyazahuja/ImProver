/-- A Lie ring is an additive group with compatible product, known as the bracket, satisfying the
Jacobi identity. -/
class LieRing (L : Type v) extends AddCommGroup L, Bracket L L where
  /-- A Lie ring bracket is additive in its first component. -/
  protected add_lie : ∀ x y z : L, ⁅x + y, z⁆ = ⁅x, z⁆ + ⁅y, z⁆
  /-- A Lie ring bracket is additive in its second component. -/
  protected lie_add : ∀ x y z : L, ⁅x, y + z⁆ = ⁅x, y⁆ + ⁅x, z⁆
  /-- A Lie ring bracket vanishes on the diagonal in L × L. -/
  protected lie_self : ∀ x : L, ⁅x, x⁆ = 0
  /-- A Lie ring bracket satisfies a Leibniz / Jacobi identity. -/
  protected leibniz_lie : ∀ x y z : L, ⁅x, ⁅y, z⁆⁆ = ⁅⁅x, y⁆, z⁆ + ⁅y, ⁅x, z⁆⁆


/-- A Lie algebra is a module with compatible product, known as the bracket, satisfying the Jacobi
identity. Forgetting the scalar multiplication, every Lie algebra is a Lie ring. -/
class LieAlgebra (R : Type u) (L : Type v) [CommRing R] [LieRing L] extends Module R L where
  /-- A Lie algebra bracket is compatible with scalar multiplication in its second argument.

  The compatibility in the first argument is not a class property, but follows since every
  Lie algebra has a natural Lie module action on itself, see `LieModule`. -/
  protected lie_smul : ∀ (t : R) (x y : L), ⁅x, t • y⁆ = t • ⁅x, y⁆


/-- A Lie ring module is an additive group, together with an additive action of a
Lie ring on this group, such that the Lie bracket acts as the commutator of endomorphisms.
(For representations of Lie *algebras* see `LieModule`.) -/
class LieRingModule (L : Type v) (M : Type w) [LieRing L] [AddCommGroup M] extends Bracket L M where
  /-- A Lie ring module bracket is additive in its first component. -/
  protected add_lie : ∀ (x y : L) (m : M), ⁅x + y, m⁆ = ⁅x, m⁆ + ⁅y, m⁆
  /-- A Lie ring module bracket is additive in its second component. -/
  protected lie_add : ∀ (x : L) (m n : M), ⁅x, m + n⁆ = ⁅x, m⁆ + ⁅x, n⁆
  /-- A Lie ring module bracket satisfies a Leibniz / Jacobi identity. -/
  protected leibniz_lie : ∀ (x y : L) (m : M), ⁅x, ⁅y, m⁆⁆ = ⁅⁅x, y⁆, m⁆ + ⁅y, ⁅x, m⁆⁆


/-- A Lie module is a module over a commutative ring, together with a linear action of a Lie
algebra on this module, such that the Lie bracket acts as the commutator of endomorphisms. -/
class LieModule (R : Type u) (L : Type v) (M : Type w) [CommRing R] [LieRing L] [LieAlgebra R L]
  [AddCommGroup M] [Module R M] [LieRingModule L M] : Prop where
  /-- A Lie module bracket is compatible with scalar multiplication in its first argument. -/
  protected smul_lie : ∀ (t : R) (x : L) (m : M), ⁅t • x, m⁆ = t • ⁅x, m⁆
  /-- A Lie module bracket is compatible with scalar multiplication in its second argument. -/
  protected lie_smul : ∀ (t : R) (x : L) (m : M), ⁅x, t • m⁆ = t • ⁅x, m⁆


/-- A tower of Lie bracket actions encapsulates the Leibniz rule for Lie bracket actions.

More precisely, it does so in a relative setting:
Let `L₁` and `L₂` be two types with Lie bracket actions on a type `M` endowed with an addition,
and additionally assume a Lie bracket action of `L₁` on `L₂`.
Then the Leibniz rule asserts for all `x : L₁`, `y : L₂`, and `m : M` that
`⁅x, ⁅y, m⁆⁆ = ⁅⁅x, y⁆, m⁆ + ⁅y, ⁅x, m⁆⁆` holds.

Common examples include the case where `L₁` is a Lie subalgebra of `L₂`
and the case where `L₂` is a Lie ideal of `L₁`. -/
class IsLieTower (L₁ L₂ M : Type*) [Bracket L₁ L₂] [Bracket L₁ M] [Bracket L₂ M] [Add M] where
  protected leibniz_lie (x : L₁) (y : L₂) (m : M) : ⁅x, ⁅y, m⁆⁆ = ⁅⁅x, y⁆, m⁆ + ⁅y, ⁅x, m⁆⁆


lemma leibniz_lie [Add M] [IsLieTower L₁ L₂ M] (x : L₁) (y : L₂) (m : M) :
    ⁅x, ⁅y, m⁆⁆ = ⁅⁅x, y⁆, m⁆ + ⁅y, ⁅x, m⁆⁆ := IsLieTower.leibniz_lie x y m


lemma lie_swap_lie [Bracket L₂ L₁] [AddCommGroup M] [IsLieTower L₁ L₂ M] [IsLieTower L₂ L₁ M]
    (x : L₁) (y : L₂) (m : M) : ⁅⁅x, y⁆, m⁆ = -⁅⁅y, x⁆, m⁆ := by
  /-
    L₁ : Type u_1
    L₂ : Type u_2
    M : Type u_3
    inst✝⁶ : Bracket L₁ L₂
    inst✝⁵ : Bracket L₁ M
    inst✝⁴ : Bracket L₂ M
    inst✝³ : Bracket L₂ L₁
    inst✝² : AddCommGroup M
    inst✝¹ : IsLieTower L₁ L₂ M
    inst✝ : IsLieTower L₂ L₁ M
    x : L₁
    y : L₂
    m : M
    ⊢ Eq (Bracket.bracket (Bracket.bracket x y) m) (Neg.neg (Bracket.bracket (Brac …
  -/
  have h1 := leibniz_lie x y m
  /-
    L₁ : Type u_1
    L₂ : Type u_2
    M : Type u_3
    inst✝⁶ : Bracket L₁ L₂
    inst✝⁵ : Bracket L₁ M
    inst✝⁴ : Bracket L₂ M
    inst✝³ : Bracket L₂ L₁
    inst✝² : AddCommGroup M
    inst✝¹ : IsLieTower L₁ L₂ M
    inst✝ : IsLieTower L₂ L₁ M
    x : L₁
    y : L₂
    m : M
    h1 : Eq (Bracket.bracket x (Bracket.bracket y m)) (HAdd.hAdd (Bracket.bracket  …
    ⊢ Eq (Bracket.bracket (Bracket.bracket x y) m) (Neg.neg (Bracket.bracket (Brac …
  -/
  have h2 := leibniz_lie y x m
  /-
    L₁ : Type u_1
    L₂ : Type u_2
    M : Type u_3
    inst✝⁶ : Bracket L₁ L₂
    inst✝⁵ : Bracket L₁ M
    inst✝⁴ : Bracket L₂ M
    inst✝³ : Bracket L₂ L₁
    inst✝² : AddCommGroup M
    inst✝¹ : IsLieTower L₁ L₂ M
    inst✝ : IsLieTower L₂ L₁ M
    x : L₁
    y : L₂
    m : M
    h1 : Eq (Bracket.bracket x (Bracket.bracket y m)) (HAdd.hAdd (Bracket.bracket  …
    h2 : Eq (Bracket.bracket y (Bracket.bracket x m)) (HAdd.hAdd (Bracket.bracket  …
    ⊢ Eq (Bracket.bracket (Bracket.bracket x y) m) (Neg.neg (Bracket.bracket (Brac …
  -/
                                            /-
                                              🎉 no goals
                                            -/
  convert congr($h1.symm - $h2) using 1 <;> simp only [add_sub_cancel_right, sub_add_cancel_right]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem add_lie : ⁅x + y, m⁆ = ⁅x, m⁆ + ⁅y, m⁆ :=
  LieRingModule.add_lie x y m


@[simp]
theorem lie_add : ⁅x, m + n⁆ = ⁅x, m⁆ + ⁅x, n⁆ :=
  LieRingModule.lie_add x m n


@[simp]
theorem smul_lie : ⁅t • x, m⁆ = t • ⁅x, m⁆ :=
  LieModule.smul_lie t x m


@[simp]
theorem lie_smul : ⁅x, t • m⁆ = t • ⁅x, m⁆ :=
  LieModule.lie_smul t x m


instance : IsLieTower L L M where
  leibniz_lie x y m := LieRingModule.leibniz_lie x y m


@[simp]
theorem lie_zero : ⁅x, 0⁆ = (0 : M) :=
  (AddMonoidHom.mk' _ (lie_add x)).map_zero


@[simp]
theorem zero_lie : ⁅(0 : L), m⁆ = 0 :=
  (AddMonoidHom.mk' (fun x : L => ⁅x, m⁆) fun x y => add_lie x y m).map_zero


@[simp]
theorem lie_self : ⁅x, x⁆ = 0 :=
  LieRing.lie_self x


instance lieRingSelfModule : LieRingModule L L :=
  { (inferInstance : LieRing L) with }


@[simp]
theorem lie_skew : -⁅y, x⁆ = ⁅x, y⁆ := by
  /-
    L : Type v
    inst✝ : LieRing L
    x y : L
    ⊢ Eq (Neg.neg (Bracket.bracket y x)) (Bracket.bracket x y)
  -/
  have h : ⁅x + y, x⁆ + ⁅x + y, y⁆ = 0 := by rw [← lie_add]; apply lie_self
  /-
    L : Type v
    inst✝ : LieRing L
    x y : L
    h : Eq (HAdd.hAdd (Bracket.bracket (HAdd.hAdd x y) x) (Bracket.bracket (HAdd.h …
    ⊢ Eq (Neg.neg (Bracket.bracket y x)) (Bracket.bracket x y)
  -/
  simpa [neg_eq_iff_add_eq_zero] using h
  /-
    🎉 no goals
  -/


/-- Every Lie algebra is a module over itself. -/
instance lieAlgebraSelfModule : LieModule R L L where
                       /-
                         R : Type u
                         L : Type v
                         M : Type w
                         N : Type w₁
                         inst✝¹⁰ : CommRing R
                         inst✝⁹ : LieRing L
                         inst✝⁸ : LieAlgebra R L
                         inst✝⁷ : AddCommGroup M
                         inst✝⁶ : Module R M
                         inst✝⁵ : LieRingModule L M
                         inst✝⁴ : LieModule R L M
                         inst✝³ : AddCommGroup N
                         inst✝² : Module R N
                         inst✝¹ : LieRingModule L N
                         inst✝ : LieModule R L N
                         t✝ : R
                         x✝ y z : L
                         m✝ n : M
                         t : R
                         x m : L
                         ⊢ Eq (Bracket.bracket (HSMul.hSMul t x) m) (HSMul.hSMul t (Bracket.bracket x m))
                       -/
  smul_lie t x m := by rw [← lie_skew, ← lie_skew x m, LieAlgebra.lie_smul, smul_neg]
                       /-
                         🎉 no goals
                       -/
                 /-
                   R : Type u
                   L : Type v
                   M : Type w
                   N : Type w₁
                   inst✝¹⁰ : CommRing R
                   inst✝⁹ : LieRing L
                   inst✝⁸ : LieAlgebra R L
                   inst✝⁷ : AddCommGroup M
                   inst✝⁶ : Module R M
                   inst✝⁵ : LieRingModule L M
                   inst✝⁴ : LieModule R L M
                   inst✝³ : AddCommGroup N
                   inst✝² : Module R N
                   inst✝¹ : LieRingModule L N
                   inst✝ : LieModule R L N
                   t : R
                   x y z : L
                   m n : M
                   ⊢ ∀ (t : R) (x m : L), Eq (Bracket.bracket x (HSMul.hSMul t m)) (HSMul.hSMul t …
                 -/
  lie_smul := by apply LieAlgebra.lie_smul
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem neg_lie : ⁅-x, m⁆ = -⁅x, m⁆ := by
  /-
    L : Type v
    M : Type w
    inst✝² : LieRing L
    inst✝¹ : AddCommGroup M
    inst✝ : LieRingModule L M
    x : L
    m : M
    ⊢ Eq (Bracket.bracket (Neg.neg x) m) (Neg.neg (Bracket.bracket x m))
  -/
  rw [← sub_eq_zero, sub_neg_eq_add, ← add_lie]
  /-
    L : Type v
    M : Type w
    inst✝² : LieRing L
    inst✝¹ : AddCommGroup M
    inst✝ : LieRingModule L M
    x : L
    m : M
    ⊢ Eq (Bracket.bracket (HAdd.hAdd (Neg.neg x) x) m) 0
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem lie_neg : ⁅x, -m⁆ = -⁅x, m⁆ := by
  /-
    L : Type v
    M : Type w
    inst✝² : LieRing L
    inst✝¹ : AddCommGroup M
    inst✝ : LieRingModule L M
    x : L
    m : M
    ⊢ Eq (Bracket.bracket x (Neg.neg m)) (Neg.neg (Bracket.bracket x m))
  -/
  rw [← sub_eq_zero, sub_neg_eq_add, ← lie_add]
  /-
    L : Type v
    M : Type w
    inst✝² : LieRing L
    inst✝¹ : AddCommGroup M
    inst✝ : LieRingModule L M
    x : L
    m : M
    ⊢ Eq (Bracket.bracket x (HAdd.hAdd (Neg.neg m) m)) 0
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
                                                     /-
                                                       L : Type v
                                                       M : Type w
                                                       inst✝² : LieRing L
                                                       inst✝¹ : AddCommGroup M
                                                       inst✝ : LieRingModule L M
                                                       x y : L
                                                       m : M
                                                       ⊢ Eq (Bracket.bracket (HSub.hSub x y) m) (HSub.hSub (Bracket.bracket x m) (Bra …
                                                     -/
theorem sub_lie : ⁅x - y, m⁆ = ⁅x, m⁆ - ⁅y, m⁆ := by simp [sub_eq_add_neg]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
                                                     /-
                                                       L : Type v
                                                       M : Type w
                                                       inst✝² : LieRing L
                                                       inst✝¹ : AddCommGroup M
                                                       inst✝ : LieRingModule L M
                                                       x : L
                                                       m n : M
                                                       ⊢ Eq (Bracket.bracket x (HSub.hSub m n)) (HSub.hSub (Bracket.bracket x m) (Bra …
                                                     -/
theorem lie_sub : ⁅x, m - n⁆ = ⁅x, m⁆ - ⁅x, n⁆ := by simp [sub_eq_add_neg]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem nsmul_lie (n : ℕ) : ⁅n • x, m⁆ = n • ⁅x, m⁆ :=
  AddMonoidHom.map_nsmul
    { toFun := fun x : L => ⁅x, m⁆, map_zero' := zero_lie m, map_add' := fun _ _ => add_lie _ _ _ }
    _ _


@[simp]
theorem lie_nsmul (n : ℕ) : ⁅x, n • m⁆ = n • ⁅x, m⁆ :=
  AddMonoidHom.map_nsmul
    { toFun := fun m : M => ⁅x, m⁆, map_zero' := lie_zero x, map_add' := fun _ _ => lie_add _ _ _}
    _ _


@[simp]
theorem zsmul_lie (a : ℤ) : ⁅a • x, m⁆ = a • ⁅x, m⁆ :=
  AddMonoidHom.map_zsmul
    { toFun := fun x : L => ⁅x, m⁆, map_zero' := zero_lie m, map_add' := fun _ _ => add_lie _ _ _ }
    _ _


@[simp]
theorem lie_zsmul (a : ℤ) : ⁅x, a • m⁆ = a • ⁅x, m⁆ :=
  AddMonoidHom.map_zsmul
    { toFun := fun m : M => ⁅x, m⁆, map_zero' := lie_zero x, map_add' := fun _ _ => lie_add _ _ _ }
    _ _


@[simp]
                                                              /-
                                                                L : Type v
                                                                M : Type w
                                                                inst✝² : LieRing L
                                                                inst✝¹ : AddCommGroup M
                                                                inst✝ : LieRingModule L M
                                                                x y : L
                                                                m : M
                                                                ⊢ Eq (Bracket.bracket (Bracket.bracket x y) m) (HSub.hSub (Bracket.bracket x ( …
                                                              -/
lemma lie_lie : ⁅⁅x, y⁆, m⁆ = ⁅x, ⁅y, m⁆⁆ - ⁅y, ⁅x, m⁆⁆ := by rw [leibniz_lie, add_sub_cancel_right]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem lie_jacobi : ⁅x, ⁅y, z⁆⁆ + ⁅y, ⁅z, x⁆⁆ + ⁅z, ⁅x, y⁆⁆ = 0 := by
  /-
    L : Type v
    inst✝ : LieRing L
    x y z : L
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Bracket.bracket x (Bracket.bracket y z)) (Bracket. …
  -/
  rw [← neg_neg ⁅x, y⁆, lie_neg z, lie_skew y x, ← lie_skew, lie_lie]
  /-
    L : Type v
    inst✝ : LieRing L
    x y z : L
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Neg.neg (HSub.hSub (Bracket.bracket y (Bracket.bra …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


instance LieRing.instLieAlgebra : LieAlgebra ℤ L where lie_smul n x y := lie_zsmul x y n


instance LinearMap.instLieRingModule : LieRingModule L (M →ₗ[R] N) where
  bracket x f :=
    { toFun := fun m => ⁅x, f m⁆ - f ⁅x, m⁆
      map_add' := fun m n => by
        /-
          R : Type u
          L : Type v
          M : Type w
          N : Type w₁
          inst✝¹⁰ : CommRing R
          inst✝⁹ : LieRing L
          inst✝⁸ : LieAlgebra R L
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : LieRingModule L M
          inst✝⁴ : LieModule R L M
          inst✝³ : AddCommGroup N
          inst✝² : Module R N
          inst✝¹ : LieRingModule L N
          inst✝ : LieModule R L N
          t : R
          x✝ y z : L
          m✝ n✝ : M
          x : L
          f : LinearMap (RingHom.id R) M N
          m n : M
          ⊢ Eq ((fun m => HSub.hSub (Bracket.bracket x (f m)) (f (Bracket.bracket x m))) …
        -/
        simp only [lie_add, LinearMap.map_add]
        /-
          R : Type u
          L : Type v
          M : Type w
          N : Type w₁
          inst✝¹⁰ : CommRing R
          inst✝⁹ : LieRing L
          inst✝⁸ : LieAlgebra R L
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : LieRingModule L M
          inst✝⁴ : LieModule R L M
          inst✝³ : AddCommGroup N
          inst✝² : Module R N
          inst✝¹ : LieRingModule L N
          inst✝ : LieModule R L N
          t : R
          x✝ y z : L
          m✝ n✝ : M
          x : L
          f : LinearMap (RingHom.id R) M N
          m n : M
          ⊢ Eq (HSub.hSub (HAdd.hAdd (Bracket.bracket x (f m)) (Bracket.bracket x (f n)) …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      map_smul' := fun t m => by
        /-
          R : Type u
          L : Type v
          M : Type w
          N : Type w₁
          inst✝¹⁰ : CommRing R
          inst✝⁹ : LieRing L
          inst✝⁸ : LieAlgebra R L
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : LieRingModule L M
          inst✝⁴ : LieModule R L M
          inst✝³ : AddCommGroup N
          inst✝² : Module R N
          inst✝¹ : LieRingModule L N
          inst✝ : LieModule R L N
          t✝ : R
          x✝ y z : L
          m✝ n : M
          x : L
          f : LinearMap (RingHom.id R) M N
          t : R
          m : M
          ⊢ Eq ({ toFun := fun m => HSub.hSub (Bracket.bracket x (f m)) (f (Bracket.brac …
        -/
        simp only [smul_sub, LinearMap.map_smul, lie_smul, RingHom.id_apply] }
        /-
          🎉 no goals
        -/
  add_lie x y f := by
    /-
      R : Type u
      L : Type v
      M : Type w
      N : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : LieRingModule L N
      inst✝ : LieModule R L N
      t : R
      x✝ y✝ z : L
      m n : M
      x y : L
      f : LinearMap (RingHom.id R) M N
      ⊢ Eq (Bracket.bracket (HAdd.hAdd x y) f) (HAdd.hAdd (Bracket.bracket x f) (Bra …
    -/
    ext n
    /-
      case h
      R : Type u
      L : Type v
      M : Type w
      N : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : LieRingModule L N
      inst✝ : LieModule R L N
      t : R
      x✝ y✝ z : L
      m n✝ : M
      x y : L
      f : LinearMap (RingHom.id R) M N
      n : M
      ⊢ Eq ((Bracket.bracket (HAdd.hAdd x y) f) n) ((HAdd.hAdd (Bracket.bracket x f) …
    -/
    simp only [add_lie, LinearMap.coe_mk, AddHom.coe_mk, LinearMap.add_apply, LinearMap.map_add]
    /-
      case h
      R : Type u
      L : Type v
      M : Type w
      N : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : LieRingModule L N
      inst✝ : LieModule R L N
      t : R
      x✝ y✝ z : L
      m n✝ : M
      x y : L
      f : LinearMap (RingHom.id R) M N
      n : M
      ⊢ Eq (HSub.hSub (HAdd.hAdd (Bracket.bracket x (f n)) (Bracket.bracket y (f n)) …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/
  lie_add x f g := by
    /-
      R : Type u
      L : Type v
      M : Type w
      N : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : LieRingModule L N
      inst✝ : LieModule R L N
      t : R
      x✝ y z : L
      m n : M
      x : L
      f g : LinearMap (RingHom.id R) M N
      ⊢ Eq (Bracket.bracket x (HAdd.hAdd f g)) (HAdd.hAdd (Bracket.bracket x f) (Bra …
    -/
    ext n
    /-
      case h
      R : Type u
      L : Type v
      M : Type w
      N : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : LieRingModule L N
      inst✝ : LieModule R L N
      t : R
      x✝ y z : L
      m n✝ : M
      x : L
      f g : LinearMap (RingHom.id R) M N
      n : M
      ⊢ Eq ((Bracket.bracket x (HAdd.hAdd f g)) n) ((HAdd.hAdd (Bracket.bracket x f) …
    -/
    simp only [LinearMap.coe_mk, AddHom.coe_mk, lie_add, LinearMap.add_apply]
    /-
      case h
      R : Type u
      L : Type v
      M : Type w
      N : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : LieRingModule L N
      inst✝ : LieModule R L N
      t : R
      x✝ y z : L
      m n✝ : M
      x : L
      f g : LinearMap (RingHom.id R) M N
      n : M
      ⊢ Eq (HSub.hSub (HAdd.hAdd (Bracket.bracket x (f n)) (Bracket.bracket x (g n)) …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/
  leibniz_lie x y f := by
    /-
      R : Type u
      L : Type v
      M : Type w
      N : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : LieRingModule L N
      inst✝ : LieModule R L N
      t : R
      x✝ y✝ z : L
      m n : M
      x y : L
      f : LinearMap (RingHom.id R) M N
      ⊢ Eq (Bracket.bracket x (Bracket.bracket y f)) (HAdd.hAdd (Bracket.bracket (Br …
    -/
    ext n
    simp only [lie_lie, LinearMap.coe_mk, AddHom.coe_mk, LinearMap.map_sub, LinearMap.add_apply,
      lie_sub]
    /-
      case h
      R : Type u
      L : Type v
      M : Type w
      N : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : LieRingModule L N
      inst✝ : LieModule R L N
      t : R
      x✝ y✝ z : L
      m n✝ : M
      x y : L
      f : LinearMap (RingHom.id R) M N
      n : M
      ⊢ Eq (HSub.hSub (HSub.hSub (Bracket.bracket x (Bracket.bracket y (f n))) (Brac …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/


@[simp]
theorem LieHom.lie_apply (f : M →ₗ[R] N) (x : L) (m : M) : ⁅x, f⁆ m = ⁅x, f m⁆ - f ⁅x, m⁆ :=
  rfl


instance LinearMap.instLieModule : LieModule R L (M →ₗ[R] N) where
  smul_lie t x f := by
    /-
      R : Type u
      L : Type v
      M : Type w
      N : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : LieRingModule L N
      inst✝ : LieModule R L N
      t✝ : R
      x✝ y z : L
      m n : M
      t : R
      x : L
      f : LinearMap (RingHom.id R) M N
      ⊢ Eq (Bracket.bracket (HSMul.hSMul t x) f) (HSMul.hSMul t (Bracket.bracket x f))
    -/
    ext n
    /-
      case h
      R : Type u
      L : Type v
      M : Type w
      N : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : LieRingModule L N
      inst✝ : LieModule R L N
      t✝ : R
      x✝ y z : L
      m n✝ : M
      t : R
      x : L
      f : LinearMap (RingHom.id R) M N
      n : M
      ⊢ Eq ((Bracket.bracket (HSMul.hSMul t x) f) n) ((HSMul.hSMul t (Bracket.bracke …
    -/
    simp only [smul_sub, smul_lie, LinearMap.smul_apply, LieHom.lie_apply, LinearMap.map_smul]
    /-
      🎉 no goals
    -/
  lie_smul t x f := by
    /-
      R : Type u
      L : Type v
      M : Type w
      N : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : LieRingModule L N
      inst✝ : LieModule R L N
      t✝ : R
      x✝ y z : L
      m n : M
      t : R
      x : L
      f : LinearMap (RingHom.id R) M N
      ⊢ Eq (Bracket.bracket x (HSMul.hSMul t f)) (HSMul.hSMul t (Bracket.bracket x f))
    -/
    ext n
    /-
      case h
      R : Type u
      L : Type v
      M : Type w
      N : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : AddCommGroup N
      inst✝² : Module R N
      inst✝¹ : LieRingModule L N
      inst✝ : LieModule R L N
      t✝ : R
      x✝ y z : L
      m n✝ : M
      t : R
      x : L
      f : LinearMap (RingHom.id R) M N
      n : M
      ⊢ Eq ((Bracket.bracket x (HSMul.hSMul t f)) n) ((HSMul.hSMul t (Bracket.bracke …
    -/
    simp only [smul_sub, LinearMap.smul_apply, LieHom.lie_apply, lie_smul]
    /-
      🎉 no goals
    -/


/-- We could avoid defining this by instead defining a `LieRingModule L R` instance with a zero
bracket and relying on `LinearMap.instLieRingModule`. We do not do this because in the case that
`L = R` we would have a non-defeq diamond via `Ring.instBracket`. -/
instance Module.Dual.instLieRingModule : LieRingModule L (M →ₗ[R] R) where
  bracket := fun x f ↦
    { toFun := fun m ↦ - f ⁅x, m⁆
                     /-
                       R : Type u
                       L : Type v
                       M : Type w
                       N : Type w₁
                       inst✝¹⁰ : CommRing R
                       inst✝⁹ : LieRing L
                       inst✝⁸ : LieAlgebra R L
                       inst✝⁷ : AddCommGroup M
                       inst✝⁶ : Module R M
                       inst✝⁵ : LieRingModule L M
                       inst✝⁴ : LieModule R L M
                       inst✝³ : AddCommGroup N
                       inst✝² : Module R N
                       inst✝¹ : LieRingModule L N
                       inst✝ : LieModule R L N
                       t : R
                       x✝ y z : L
                       m n : M
                       x : L
                       f : LinearMap (RingHom.id R) M R
                       ⊢ ∀ (x_1 y : M), Eq ((fun m => Neg.neg (f (Bracket.bracket x m))) (HAdd.hAdd x …
                     -/
      map_add' := by simp [-neg_add_rev, neg_add]
                     /-
                       🎉 no goals
                     -/
                      /-
                        R : Type u
                        L : Type v
                        M : Type w
                        N : Type w₁
                        inst✝¹⁰ : CommRing R
                        inst✝⁹ : LieRing L
                        inst✝⁸ : LieAlgebra R L
                        inst✝⁷ : AddCommGroup M
                        inst✝⁶ : Module R M
                        inst✝⁵ : LieRingModule L M
                        inst✝⁴ : LieModule R L M
                        inst✝³ : AddCommGroup N
                        inst✝² : Module R N
                        inst✝¹ : LieRingModule L N
                        inst✝ : LieModule R L N
                        t : R
                        x✝ y z : L
                        m n : M
                        x : L
                        f : LinearMap (RingHom.id R) M R
                        ⊢ ∀ (m : R) (x_1 : M), Eq ({ toFun := fun m => Neg.neg (f (Bracket.bracket x m …
                      -/
      map_smul' := by simp }
                      /-
                        🎉 no goals
                      -/
                            /-
                              R : Type u
                              L : Type v
                              M : Type w
                              N : Type w₁
                              inst✝¹⁰ : CommRing R
                              inst✝⁹ : LieRing L
                              inst✝⁸ : LieAlgebra R L
                              inst✝⁷ : AddCommGroup M
                              inst✝⁶ : Module R M
                              inst✝⁵ : LieRingModule L M
                              inst✝⁴ : LieModule R L M
                              inst✝³ : AddCommGroup N
                              inst✝² : Module R N
                              inst✝¹ : LieRingModule L N
                              inst✝ : LieModule R L N
                              t : R
                              x✝ y✝ z : L
                              m✝ n : M
                              x y : L
                              m : LinearMap (RingHom.id R) M R
                              ⊢ Eq (Bracket.bracket (HAdd.hAdd x y) m) (HAdd.hAdd (Bracket.bracket x m) (Bra …
                            -/
  add_lie := fun x y m ↦ by ext n; simp [-neg_add_rev, neg_add]
                                   /-
                                     🎉 no goals
                                   -/
                            /-
                              R : Type u
                              L : Type v
                              M : Type w
                              N : Type w₁
                              inst✝¹⁰ : CommRing R
                              inst✝⁹ : LieRing L
                              inst✝⁸ : LieAlgebra R L
                              inst✝⁷ : AddCommGroup M
                              inst✝⁶ : Module R M
                              inst✝⁵ : LieRingModule L M
                              inst✝⁴ : LieModule R L M
                              inst✝³ : AddCommGroup N
                              inst✝² : Module R N
                              inst✝¹ : LieRingModule L N
                              inst✝ : LieModule R L N
                              t : R
                              x✝ y z : L
                              m✝ n✝ : M
                              x : L
                              m n : LinearMap (RingHom.id R) M R
                              ⊢ Eq (Bracket.bracket x (HAdd.hAdd m n)) (HAdd.hAdd (Bracket.bracket x m) (Bra …
                            -/
  lie_add := fun x m n ↦ by ext p; simp [-neg_add_rev, neg_add]
                                   /-
                                     🎉 no goals
                                   -/
                                /-
                                  R : Type u
                                  L : Type v
                                  M : Type w
                                  N : Type w₁
                                  inst✝¹⁰ : CommRing R
                                  inst✝⁹ : LieRing L
                                  inst✝⁸ : LieAlgebra R L
                                  inst✝⁷ : AddCommGroup M
                                  inst✝⁶ : Module R M
                                  inst✝⁵ : LieRingModule L M
                                  inst✝⁴ : LieModule R L M
                                  inst✝³ : AddCommGroup N
                                  inst✝² : Module R N
                                  inst✝¹ : LieRingModule L N
                                  inst✝ : LieModule R L N
                                  t : R
                                  x✝ y z : L
                                  m✝ n✝ : M
                                  x m : L
                                  n : LinearMap (RingHom.id R) M R
                                  ⊢ Eq (Bracket.bracket x (Bracket.bracket m n)) (HAdd.hAdd (Bracket.bracket (Br …
                                -/
  leibniz_lie := fun x m n ↦ by ext p; simp
                                       /-
                                         🎉 no goals
                                       -/


@[simp] lemma Module.Dual.lie_apply (f : M →ₗ[R] R) : ⁅x, f⁆ m = - f ⁅x, m⁆ := rfl


instance Module.Dual.instLieModule : LieModule R L (M →ₗ[R] R) where
                             /-
                               R : Type u
                               L : Type v
                               M : Type w
                               N : Type w₁
                               inst✝¹⁰ : CommRing R
                               inst✝⁹ : LieRing L
                               inst✝⁸ : LieAlgebra R L
                               inst✝⁷ : AddCommGroup M
                               inst✝⁶ : Module R M
                               inst✝⁵ : LieRingModule L M
                               inst✝⁴ : LieModule R L M
                               inst✝³ : AddCommGroup N
                               inst✝² : Module R N
                               inst✝¹ : LieRingModule L N
                               inst✝ : LieModule R L N
                               t✝ : R
                               x✝ y z : L
                               m✝ n : M
                               t : R
                               x : L
                               m : LinearMap (RingHom.id R) M R
                               ⊢ Eq (Bracket.bracket (HSMul.hSMul t x) m) (HSMul.hSMul t (Bracket.bracket x m))
                             -/
  smul_lie := fun t x m ↦ by ext n; simp
                                    /-
                                      🎉 no goals
                                    -/
                             /-
                               R : Type u
                               L : Type v
                               M : Type w
                               N : Type w₁
                               inst✝¹⁰ : CommRing R
                               inst✝⁹ : LieRing L
                               inst✝⁸ : LieAlgebra R L
                               inst✝⁷ : AddCommGroup M
                               inst✝⁶ : Module R M
                               inst✝⁵ : LieRingModule L M
                               inst✝⁴ : LieModule R L M
                               inst✝³ : AddCommGroup N
                               inst✝² : Module R N
                               inst✝¹ : LieRingModule L N
                               inst✝ : LieModule R L N
                               t✝ : R
                               x✝ y z : L
                               m✝ n : M
                               t : R
                               x : L
                               m : LinearMap (RingHom.id R) M R
                               ⊢ Eq (Bracket.bracket x (HSMul.hSMul t m)) (HSMul.hSMul t (Bracket.bracket x m))
                             -/
  lie_smul := fun t x m ↦ by ext n; simp
                                    /-
                                      🎉 no goals
                                    -/


/-- A morphism of Lie algebras is a linear map respecting the bracket operations. -/
structure LieHom (R L L' : Type*) [CommRing R] [LieRing L] [LieAlgebra R L]
  [LieRing L'] [LieAlgebra R L'] extends L →ₗ[R] L' where
  /-- A morphism of Lie algebras is compatible with brackets. -/
  map_lie' : ∀ {x y : L}, toFun ⁅x, y⁆ = ⁅toFun x, toFun y⁆


@[inherit_doc]
notation:25 L " →ₗ⁅" R:25 "⁆ " L':0 => LieHom R L L'


instance : Coe (L₁ →ₗ⁅R⁆ L₂) (L₁ →ₗ[R] L₂) :=
  ⟨LieHom.toLinearMap⟩


instance : FunLike (L₁ →ₗ⁅R⁆ L₂) L₁ L₂ where
  coe f := f.toFun
  coe_injective' x y h := by
    /-
      R : Type u
      L₁ : Type v
      L₂ : Type w
      L₃ : Type w₁
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L₁
      inst✝⁴ : LieAlgebra R L₁
      inst✝³ : LieRing L₂
      inst✝² : LieAlgebra R L₂
      inst✝¹ : LieRing L₃
      inst✝ : LieAlgebra R L₃
      x y : LieHom R L₁ L₂
      h : Eq ((fun f => (↑f).toFun) x) ((fun f => (↑f).toFun) y)
      ⊢ Eq x y
    -/
    cases x; cases y; simp at h; simp [h]
                                 /-
                                   🎉 no goals
                                 -/


@[simp, norm_cast]
theorem coe_toLinearMap (f : L₁ →ₗ⁅R⁆ L₂) : ⇑(f : L₁ →ₗ[R] L₂) = f :=
  rfl


@[simp]
theorem toFun_eq_coe (f : L₁ →ₗ⁅R⁆ L₂) : f.toFun = ⇑f :=
  rfl


@[simp]
theorem map_smul (f : L₁ →ₗ⁅R⁆ L₂) (c : R) (x : L₁) : f (c • x) = c • f x :=
  LinearMap.map_smul (f : L₁ →ₗ[R] L₂) c x


@[simp]
theorem map_add (f : L₁ →ₗ⁅R⁆ L₂) (x y : L₁) : f (x + y) = f x + f y :=
  LinearMap.map_add (f : L₁ →ₗ[R] L₂) x y


@[simp]
theorem map_sub (f : L₁ →ₗ⁅R⁆ L₂) (x y : L₁) : f (x - y) = f x - f y :=
  LinearMap.map_sub (f : L₁ →ₗ[R] L₂) x y


@[simp]
theorem map_neg (f : L₁ →ₗ⁅R⁆ L₂) (x : L₁) : f (-x) = -f x :=
  LinearMap.map_neg (f : L₁ →ₗ[R] L₂) x


@[simp]
theorem map_lie (f : L₁ →ₗ⁅R⁆ L₂) (x y : L₁) : f ⁅x, y⁆ = ⁅f x, f y⁆ :=
  LieHom.map_lie' f


@[simp]
theorem map_zero (f : L₁ →ₗ⁅R⁆ L₂) : f 0 = 0 :=
  (f : L₁ →ₗ[R] L₂).map_zero


/-- The identity map is a morphism of Lie algebras. -/
def id : L₁ →ₗ⁅R⁆ L₁ :=
  { (LinearMap.id : L₁ →ₗ[R] L₁) with map_lie' := rfl }


@[simp]
theorem coe_id : ⇑(id : L₁ →ₗ⁅R⁆ L₁) = _root_.id :=
  rfl


theorem id_apply (x : L₁) : (id : L₁ →ₗ⁅R⁆ L₁) x = x :=
  rfl


/-- The constant 0 map is a Lie algebra morphism. -/
instance : Zero (L₁ →ₗ⁅R⁆ L₂) :=
                                           /-
                                             R : Type u
                                             L₁ : Type v
                                             L₂ : Type w
                                             L₃ : Type w₁
                                             inst✝⁶ : CommRing R
                                             inst✝⁵ : LieRing L₁
                                             inst✝⁴ : LieAlgebra R L₁
                                             inst✝³ : LieRing L₂
                                             inst✝² : LieAlgebra R L₂
                                             inst✝¹ : LieRing L₃
                                             inst✝ : LieAlgebra R L₃
                                             ⊢ ∀ {x y : L₁}, Eq (__src✝.toFun (Bracket.bracket x y)) (Bracket.bracket (__sr …
                                           -/
  ⟨{ (0 : L₁ →ₗ[R] L₂) with map_lie' := by simp }⟩
                                           /-
                                             🎉 no goals
                                           -/


@[norm_cast, simp]
theorem coe_zero : ((0 : L₁ →ₗ⁅R⁆ L₂) : L₁ → L₂) = 0 :=
  rfl


theorem zero_apply (x : L₁) : (0 : L₁ →ₗ⁅R⁆ L₂) x = 0 :=
  rfl


/-- The identity map is a Lie algebra morphism. -/
instance : One (L₁ →ₗ⁅R⁆ L₁) :=
  ⟨id⟩


@[simp]
theorem coe_one : ((1 : L₁ →ₗ⁅R⁆ L₁) : L₁ → L₁) = _root_.id :=
  rfl


theorem one_apply (x : L₁) : (1 : L₁ →ₗ⁅R⁆ L₁) x = x :=
  rfl


instance : Inhabited (L₁ →ₗ⁅R⁆ L₂) :=
  ⟨0⟩


theorem coe_injective : @Function.Injective (L₁ →ₗ⁅R⁆ L₂) (L₁ → L₂) (↑) := by
  /-
    R : Type u
    L₁ : Type v
    L₂ : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L₁
    inst✝² : LieAlgebra R L₁
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    ⊢ Function.Injective DFunLike.coe
  -/
  rintro ⟨⟨⟨f, _⟩, _⟩, _⟩ ⟨⟨⟨g, _⟩, _⟩, _⟩ h
  /-
    case mk.mk.mk.mk.mk.mk
    R : Type u
    L₁ : Type v
    L₂ : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L₁
    inst✝² : LieAlgebra R L₁
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : L₁ → L₂
    map_add'✝¹ : ∀ (x y : L₁), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    map_smul'✝¹ : ∀ (m : R) (x : L₁), Eq ({ toFun := f, map_add' := map_add'✝¹ }.t …
    map_lie'✝¹ : ∀ {x y : L₁}, Eq ({ toFun := f, map_add' := map_add'✝¹, map_smul' …
    g : L₁ → L₂
    map_add'✝ : ∀ (x y : L₁), Eq (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    map_smul'✝ : ∀ (m : R) (x : L₁), Eq ({ toFun := g, map_add' := map_add'✝ }.toF …
    map_lie'✝ : ∀ {x y : L₁}, Eq ({ toFun := g, map_add' := map_add'✝, map_smul' : …
    h : Eq ⇑{ toFun := f, map_add' := map_add'✝¹, map_smul' := map_smul'✝¹, map_li …
    ⊢ Eq { toFun := f, map_add' := map_add'✝¹, map_smul' := map_smul'✝¹, map_lie'  …
  -/
  congr
  /-
    🎉 no goals
  -/


@[ext]
theorem ext {f g : L₁ →ₗ⁅R⁆ L₂} (h : ∀ x, f x = g x) : f = g :=
  coe_injective <| funext h


theorem congr_fun {f g : L₁ →ₗ⁅R⁆ L₂} (h : f = g) (x : L₁) : f x = g x :=
  h ▸ rfl


@[simp]
theorem mk_coe (f : L₁ →ₗ⁅R⁆ L₂) (h₁ h₂ h₃) : (⟨⟨⟨f, h₁⟩, h₂⟩, h₃⟩ : L₁ →ₗ⁅R⁆ L₂) = f := by
  /-
    R : Type u
    L₁ : Type v
    L₂ : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L₁
    inst✝² : LieAlgebra R L₁
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L₁ L₂
    h₁ : ∀ (x y : L₁), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    h₂ : ∀ (m : R) (x : L₁), Eq ({ toFun := ⇑f, map_add' := h₁ }.toFun (HSMul.hSMu …
    h₃ : ∀ {x y : L₁}, Eq ({ toFun := ⇑f, map_add' := h₁, map_smul' := h₂ }.toFun  …
    ⊢ Eq { toFun := ⇑f, map_add' := h₁, map_smul' := h₂, map_lie' := h₃ } f
  -/
  ext
  /-
    case h
    R : Type u
    L₁ : Type v
    L₂ : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L₁
    inst✝² : LieAlgebra R L₁
    inst✝¹ : LieRing L₂
    inst✝ : LieAlgebra R L₂
    f : LieHom R L₁ L₂
    h₁ : ∀ (x y : L₁), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    h₂ : ∀ (m : R) (x : L₁), Eq ({ toFun := ⇑f, map_add' := h₁ }.toFun (HSMul.hSMu …
    h₃ : ∀ {x y : L₁}, Eq ({ toFun := ⇑f, map_add' := h₁, map_smul' := h₂ }.toFun  …
    x✝ : L₁
    ⊢ Eq ({ toFun := ⇑f, map_add' := h₁, map_smul' := h₂, map_lie' := h₃ } x✝) (f  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_mk (f : L₁ → L₂) (h₁ h₂ h₃) : ((⟨⟨⟨f, h₁⟩, h₂⟩, h₃⟩ : L₁ →ₗ⁅R⁆ L₂) : L₁ → L₂) = f :=
  rfl


/-- The composition of morphisms is a morphism. -/
def comp (f : L₂ →ₗ⁅R⁆ L₃) (g : L₁ →ₗ⁅R⁆ L₂) : L₁ →ₗ⁅R⁆ L₃ :=
  { LinearMap.comp f.toLinearMap g.toLinearMap with
    map_lie' := by
      /-
        R : Type u
        L₁ : Type v
        L₂ : Type w
        L₃ : Type w₁
        inst✝⁶ : CommRing R
        inst✝⁵ : LieRing L₁
        inst✝⁴ : LieAlgebra R L₁
        inst✝³ : LieRing L₂
        inst✝² : LieAlgebra R L₂
        inst✝¹ : LieRing L₃
        inst✝ : LieAlgebra R L₃
        f : LieHom R L₂ L₃
        g : LieHom R L₁ L₂
        ⊢ ∀ {x y : L₁}, Eq (__src✝.toFun (Bracket.bracket x y)) (Bracket.bracket (__sr …
      -/
      intros x y
      /-
        R : Type u
        L₁ : Type v
        L₂ : Type w
        L₃ : Type w₁
        inst✝⁶ : CommRing R
        inst✝⁵ : LieRing L₁
        inst✝⁴ : LieAlgebra R L₁
        inst✝³ : LieRing L₂
        inst✝² : LieAlgebra R L₂
        inst✝¹ : LieRing L₃
        inst✝ : LieAlgebra R L₃
        f : LieHom R L₂ L₃
        g : LieHom R L₁ L₂
        x y : L₁
        ⊢ Eq (__src✝.toFun (Bracket.bracket x y)) (Bracket.bracket (__src✝.toFun x) (_ …
      -/
      change f (g ⁅x, y⁆) = ⁅f (g x), f (g y)⁆
      /-
        R : Type u
        L₁ : Type v
        L₂ : Type w
        L₃ : Type w₁
        inst✝⁶ : CommRing R
        inst✝⁵ : LieRing L₁
        inst✝⁴ : LieAlgebra R L₁
        inst✝³ : LieRing L₂
        inst✝² : LieAlgebra R L₂
        inst✝¹ : LieRing L₃
        inst✝ : LieAlgebra R L₃
        f : LieHom R L₂ L₃
        g : LieHom R L₁ L₂
        x y : L₁
        ⊢ Eq (f (g (Bracket.bracket x y))) (Bracket.bracket (f (g x)) (f (g y)))
      -/
      rw [map_lie, map_lie] }
      /-
        🎉 no goals
      -/


theorem comp_apply (f : L₂ →ₗ⁅R⁆ L₃) (g : L₁ →ₗ⁅R⁆ L₂) (x : L₁) : f.comp g x = f (g x) :=
  rfl


@[norm_cast, simp]
theorem coe_comp (f : L₂ →ₗ⁅R⁆ L₃) (g : L₁ →ₗ⁅R⁆ L₂) : (f.comp g : L₁ → L₃) = f ∘ g :=
  rfl


@[norm_cast, simp]
theorem toLinearMap_comp (f : L₂ →ₗ⁅R⁆ L₃) (g : L₁ →ₗ⁅R⁆ L₂) :
    (f.comp g : L₁ →ₗ[R] L₃) = (f : L₂ →ₗ[R] L₃).comp (g : L₁ →ₗ[R] L₂) :=
  rfl


@[deprecated (since := "2024-12-30")] alias coe_linearMap_comp := toLinearMap_comp


@[simp]
theorem comp_id (f : L₁ →ₗ⁅R⁆ L₂) : f.comp (id : L₁ →ₗ⁅R⁆ L₁) = f :=
  rfl


@[simp]
theorem id_comp (f : L₁ →ₗ⁅R⁆ L₂) : (id : L₂ →ₗ⁅R⁆ L₂).comp f = f :=
  rfl


/-- The inverse of a bijective morphism is a morphism. -/
def inverse (f : L₁ →ₗ⁅R⁆ L₂) (g : L₂ → L₁) (h₁ : Function.LeftInverse g f)
    (h₂ : Function.RightInverse g f) : L₂ →ₗ⁅R⁆ L₁ :=
  { LinearMap.inverse f.toLinearMap g h₁ h₂ with
    map_lie' := by
      /-
        R : Type u
        L₁ : Type v
        L₂ : Type w
        L₃ : Type w₁
        inst✝⁶ : CommRing R
        inst✝⁵ : LieRing L₁
        inst✝⁴ : LieAlgebra R L₁
        inst✝³ : LieRing L₂
        inst✝² : LieAlgebra R L₂
        inst✝¹ : LieRing L₃
        inst✝ : LieAlgebra R L₃
        f : LieHom R L₁ L₂
        g : L₂ → L₁
        h₁ : Function.LeftInverse g ⇑f
        h₂ : Function.RightInverse g ⇑f
        ⊢ ∀ {x y : L₂}, Eq (__src✝.toFun (Bracket.bracket x y)) (Bracket.bracket (__sr …
      -/
      intros x y
      calc
        g ⁅x, y⁆ = g ⁅f (g x), f (g y)⁆ := by conv_lhs => rw [← h₂ x, ← h₂ y]
        _ = g (f ⁅g x, g y⁆) := by rw [map_lie]
        _ = ⁅g x, g y⁆ := h₁ _
         }


/-- A Lie ring module may be pulled back along a morphism of Lie algebras.

See note [reducible non-instances]. -/
def LieRingModule.compLieHom : LieRingModule L₁ M where
  bracket x m := ⁅f x, m⁆
  lie_add x := lie_add (f x)
                      /-
                        R : Type u
                        L₁ : Type v
                        L₂ : Type w
                        M : Type w₁
                        inst✝⁶ : CommRing R
                        inst✝⁵ : LieRing L₁
                        inst✝⁴ : LieAlgebra R L₁
                        inst✝³ : LieRing L₂
                        inst✝² : LieAlgebra R L₂
                        inst✝¹ : AddCommGroup M
                        inst✝ : LieRingModule L₂ M
                        f : LieHom R L₁ L₂
                        x y : L₁
                        m : M
                        ⊢ Eq (Bracket.bracket (HAdd.hAdd x y) m) (HAdd.hAdd (Bracket.bracket x m) (Bra …
                      -/
  add_lie x y m := by simp only [LieHom.map_add, add_lie]
                      /-
                        🎉 no goals
                      -/
                          /-
                            R : Type u
                            L₁ : Type v
                            L₂ : Type w
                            M : Type w₁
                            inst✝⁶ : CommRing R
                            inst✝⁵ : LieRing L₁
                            inst✝⁴ : LieAlgebra R L₁
                            inst✝³ : LieRing L₂
                            inst✝² : LieAlgebra R L₂
                            inst✝¹ : AddCommGroup M
                            inst✝ : LieRingModule L₂ M
                            f : LieHom R L₁ L₂
                            x y : L₁
                            m : M
                            ⊢ Eq (Bracket.bracket x (Bracket.bracket y m)) (HAdd.hAdd (Bracket.bracket (Br …
                          -/
  leibniz_lie x y m := by simp only [lie_lie, sub_add_cancel, LieHom.map_lie]
                          /-
                            🎉 no goals
                          -/


theorem LieRingModule.compLieHom_apply (x : L₁) (m : M) :
    haveI := LieRingModule.compLieHom M f
    ⁅x, m⁆ = ⁅f x, m⁆ :=
  rfl


/-- A Lie module may be pulled back along a morphism of Lie algebras. -/
theorem LieModule.compLieHom [Module R M] [LieModule R L₂ M] :
    @LieModule R L₁ M _ _ _ _ _ (LieRingModule.compLieHom M f) :=
  { __ := LieRingModule.compLieHom M f
    smul_lie := fun t x m => by
      /-
        R : Type u
        L₁ : Type v
        L₂ : Type w
        M : Type w₁
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L₁
        inst✝⁶ : LieAlgebra R L₁
        inst✝⁵ : LieRing L₂
        inst✝⁴ : LieAlgebra R L₂
        inst✝³ : AddCommGroup M
        inst✝² : LieRingModule L₂ M
        f : LieHom R L₁ L₂
        inst✝¹ : Module R M
        inst✝ : LieModule R L₂ M
        t : R
        x : L₁
        m : M
        ⊢ Eq (Bracket.bracket (HSMul.hSMul t x) m) (HSMul.hSMul t (Bracket.bracket x m))
      -/
      simp only [LieRingModule.compLieHom_apply, smul_lie, LieHom.map_smul]
      /-
        🎉 no goals
      -/
    lie_smul := fun t x m => by
      /-
        R : Type u
        L₁ : Type v
        L₂ : Type w
        M : Type w₁
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L₁
        inst✝⁶ : LieAlgebra R L₁
        inst✝⁵ : LieRing L₂
        inst✝⁴ : LieAlgebra R L₂
        inst✝³ : AddCommGroup M
        inst✝² : LieRingModule L₂ M
        f : LieHom R L₁ L₂
        inst✝¹ : Module R M
        inst✝ : LieModule R L₂ M
        t : R
        x : L₁
        m : M
        ⊢ Eq (Bracket.bracket x (HSMul.hSMul t m)) (HSMul.hSMul t (Bracket.bracket x m))
      -/
      simp only [LieRingModule.compLieHom_apply, lie_smul] }
      /-
        🎉 no goals
      -/


/-- An equivalence of Lie algebras is a morphism which is also a linear equivalence. We could
instead define an equivalence to be a morphism which is also a (plain) equivalence. However it is
more convenient to define via linear equivalence to get `.toLinearEquiv` for free. -/
structure LieEquiv (R : Type u) (L : Type v) (L' : Type w) [CommRing R] [LieRing L] [LieAlgebra R L]
  [LieRing L'] [LieAlgebra R L'] extends L →ₗ⁅R⁆ L' where
  /-- The inverse function of an equivalence of Lie algebras -/
  invFun : L' → L
  /-- The inverse function of an equivalence of Lie algebras is a left inverse of the underlying
  function. -/
  left_inv : Function.LeftInverse invFun toLieHom.toFun
  /-- The inverse function of an equivalence of Lie algebras is a right inverse of the underlying
  function. -/
  right_inv : Function.RightInverse invFun toLieHom.toFun


@[inherit_doc]
notation:50 L " ≃ₗ⁅" R "⁆ " L' => LieEquiv R L L'


/-- Consider an equivalence of Lie algebras as a linear equivalence. -/
def toLinearEquiv (f : L₁ ≃ₗ⁅R⁆ L₂) : L₁ ≃ₗ[R] L₂ :=
  { f.toLieHom, f with }


instance hasCoeToLieHom : Coe (L₁ ≃ₗ⁅R⁆ L₂) (L₁ →ₗ⁅R⁆ L₂) :=
  ⟨toLieHom⟩


instance hasCoeToLinearEquiv : Coe (L₁ ≃ₗ⁅R⁆ L₂) (L₁ ≃ₗ[R] L₂) :=
  ⟨toLinearEquiv⟩


instance : EquivLike (L₁ ≃ₗ⁅R⁆ L₂) L₁ L₂ where
  coe f := f.toFun
  inv f := f.invFun
  left_inv f := f.left_inv
  right_inv f := f.right_inv
                                 /-
                                   R : Type u
                                   L₁ : Type v
                                   L₂ : Type w
                                   L₃ : Type w₁
                                   inst✝⁶ : CommRing R
                                   inst✝⁵ : LieRing L₁
                                   inst✝⁴ : LieRing L₂
                                   inst✝³ : LieRing L₃
                                   inst✝² : LieAlgebra R L₁
                                   inst✝¹ : LieAlgebra R L₂
                                   inst✝ : LieAlgebra R L₃
                                   f g : LieEquiv R L₁ L₂
                                   h₁ : Eq ((fun f => (↑f.toLieHom).toFun) f) ((fun f => (↑f.toLieHom).toFun) g)
                                   h₂ : Eq ((fun f => f.invFun) f) ((fun f => f.invFun) g)
                                   ⊢ Eq f g
                                 -/
  coe_injective' f g h₁ h₂ := by cases f; cases g; simp at h₁ h₂; simp [*]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem coe_toLieHom (e : L₁ ≃ₗ⁅R⁆ L₂) : ⇑(e : L₁ →ₗ⁅R⁆ L₂) = e :=
  rfl


@[deprecated (since := "2024-12-30")] alias coe_to_lieHom := coe_toLieHom


@[simp]
theorem coe_toLinearEquiv (e : L₁ ≃ₗ⁅R⁆ L₂) : ⇑(e : L₁ ≃ₗ[R] L₂) = e :=
  rfl


@[deprecated (since := "2024-12-30")] alias coe_to_linearEquiv := coe_toLinearEquiv


@[simp]
theorem toLinearEquiv_mk (f : L₁ →ₗ⁅R⁆ L₂) (g h₁ h₂) :
    (mk f g h₁ h₂ : L₁ ≃ₗ[R] L₂) =
      { f with
        invFun := g
        left_inv := h₁
        right_inv := h₂ } :=
  rfl


@[deprecated (since := "2024-12-30")] alias to_linearEquiv_mk := toLinearEquiv_mk


theorem toLinearEquiv_injective : Injective ((↑) : (L₁ ≃ₗ⁅R⁆ L₂) → L₁ ≃ₗ[R] L₂) := by
  /-
    R : Type u
    L₁ : Type v
    L₂ : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L₁
    inst✝² : LieRing L₂
    inst✝¹ : LieAlgebra R L₁
    inst✝ : LieAlgebra R L₂
    ⊢ Function.Injective LieEquiv.toLinearEquiv
  -/
  rintro ⟨⟨⟨⟨f, -⟩, -⟩, -⟩, f_inv⟩ ⟨⟨⟨⟨g, -⟩, -⟩, -⟩, g_inv⟩
  /-
    case mk.mk.mk.mk.mk.mk.mk.mk
    R : Type u
    L₁ : Type v
    L₂ : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L₁
    inst✝² : LieRing L₂
    inst✝¹ : LieAlgebra R L₁
    inst✝ : LieAlgebra R L₂
    f_inv : L₂ → L₁
    f : L₁ → L₂
    map_add'✝¹ : ∀ (x y : L₁), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    map_smul'✝¹ : ∀ (m : R) (x : L₁), Eq ({ toFun := f, map_add' := map_add'✝¹ }.t …
    map_lie'✝¹ : ∀ {x y : L₁}, Eq ({ toFun := f, map_add' := map_add'✝¹, map_smul' …
    left_inv✝¹ : Function.LeftInverse f_inv (↑{ toFun := f, map_add' := map_add'✝¹ …
    right_inv✝¹ : Function.RightInverse f_inv (↑{ toFun := f, map_add' := map_add' …
    g_inv : L₂ → L₁
    g : L₁ → L₂
    map_add'✝ : ∀ (x y : L₁), Eq (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    map_smul'✝ : ∀ (m : R) (x : L₁), Eq ({ toFun := g, map_add' := map_add'✝ }.toF …
    map_lie'✝ : ∀ {x y : L₁}, Eq ({ toFun := g, map_add' := map_add'✝, map_smul' : …
    left_inv✝ : Function.LeftInverse g_inv (↑{ toFun := g, map_add' := map_add'✝,  …
    right_inv✝ : Function.RightInverse g_inv (↑{ toFun := g, map_add' := map_add'✝ …
    ⊢ Eq { toFun := f, map_add' := map_add'✝¹, map_smul' := map_smul'✝¹, map_lie'  …
  -/
  intro h
  /-
    case mk.mk.mk.mk.mk.mk.mk.mk
    R : Type u
    L₁ : Type v
    L₂ : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L₁
    inst✝² : LieRing L₂
    inst✝¹ : LieAlgebra R L₁
    inst✝ : LieAlgebra R L₂
    f_inv : L₂ → L₁
    f : L₁ → L₂
    map_add'✝¹ : ∀ (x y : L₁), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    map_smul'✝¹ : ∀ (m : R) (x : L₁), Eq ({ toFun := f, map_add' := map_add'✝¹ }.t …
    map_lie'✝¹ : ∀ {x y : L₁}, Eq ({ toFun := f, map_add' := map_add'✝¹, map_smul' …
    left_inv✝¹ : Function.LeftInverse f_inv (↑{ toFun := f, map_add' := map_add'✝¹ …
    right_inv✝¹ : Function.RightInverse f_inv (↑{ toFun := f, map_add' := map_add' …
    g_inv : L₂ → L₁
    g : L₁ → L₂
    map_add'✝ : ∀ (x y : L₁), Eq (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    map_smul'✝ : ∀ (m : R) (x : L₁), Eq ({ toFun := g, map_add' := map_add'✝ }.toF …
    map_lie'✝ : ∀ {x y : L₁}, Eq ({ toFun := g, map_add' := map_add'✝, map_smul' : …
    left_inv✝ : Function.LeftInverse g_inv (↑{ toFun := g, map_add' := map_add'✝,  …
    right_inv✝ : Function.RightInverse g_inv (↑{ toFun := g, map_add' := map_add'✝ …
    h : Eq { toFun := f, map_add' := map_add'✝¹, map_smul' := map_smul'✝¹, map_lie …
    ⊢ Eq { toFun := f, map_add' := map_add'✝¹, map_smul' := map_smul'✝¹, map_lie'  …
  -/
  simp only [toLinearEquiv_mk, LinearEquiv.mk.injEq, LinearMap.mk.injEq, AddHom.mk.injEq] at h
  /-
    case mk.mk.mk.mk.mk.mk.mk.mk
    R : Type u
    L₁ : Type v
    L₂ : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L₁
    inst✝² : LieRing L₂
    inst✝¹ : LieAlgebra R L₁
    inst✝ : LieAlgebra R L₂
    f_inv : L₂ → L₁
    f : L₁ → L₂
    map_add'✝¹ : ∀ (x y : L₁), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    map_smul'✝¹ : ∀ (m : R) (x : L₁), Eq ({ toFun := f, map_add' := map_add'✝¹ }.t …
    map_lie'✝¹ : ∀ {x y : L₁}, Eq ({ toFun := f, map_add' := map_add'✝¹, map_smul' …
    left_inv✝¹ : Function.LeftInverse f_inv (↑{ toFun := f, map_add' := map_add'✝¹ …
    right_inv✝¹ : Function.RightInverse f_inv (↑{ toFun := f, map_add' := map_add' …
    g_inv : L₂ → L₁
    g : L₁ → L₂
    map_add'✝ : ∀ (x y : L₁), Eq (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    map_smul'✝ : ∀ (m : R) (x : L₁), Eq ({ toFun := g, map_add' := map_add'✝ }.toF …
    map_lie'✝ : ∀ {x y : L₁}, Eq ({ toFun := g, map_add' := map_add'✝, map_smul' : …
    left_inv✝ : Function.LeftInverse g_inv (↑{ toFun := g, map_add' := map_add'✝,  …
    right_inv✝ : Function.RightInverse g_inv (↑{ toFun := g, map_add' := map_add'✝ …
    h : And (Eq f g) (Eq f_inv g_inv)
    ⊢ Eq { toFun := f, map_add' := map_add'✝¹, map_smul' := map_smul'✝¹, map_lie'  …
  -/
  congr
  /-
    case mk.mk.mk.mk.mk.mk.mk.mk.e_toLieHom.e_toLinearMap.e_toAddHom.e_toFun
    R : Type u
    L₁ : Type v
    L₂ : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L₁
    inst✝² : LieRing L₂
    inst✝¹ : LieAlgebra R L₁
    inst✝ : LieAlgebra R L₂
    f_inv : L₂ → L₁
    f : L₁ → L₂
    map_add'✝¹ : ∀ (x y : L₁), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    map_smul'✝¹ : ∀ (m : R) (x : L₁), Eq ({ toFun := f, map_add' := map_add'✝¹ }.t …
    map_lie'✝¹ : ∀ {x y : L₁}, Eq ({ toFun := f, map_add' := map_add'✝¹, map_smul' …
    left_inv✝¹ : Function.LeftInverse f_inv (↑{ toFun := f, map_add' := map_add'✝¹ …
    right_inv✝¹ : Function.RightInverse f_inv (↑{ toFun := f, map_add' := map_add' …
    g_inv : L₂ → L₁
    g : L₁ → L₂
    map_add'✝ : ∀ (x y : L₁), Eq (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    map_smul'✝ : ∀ (m : R) (x : L₁), Eq ({ toFun := g, map_add' := map_add'✝ }.toF …
    map_lie'✝ : ∀ {x y : L₁}, Eq ({ toFun := g, map_add' := map_add'✝, map_smul' : …
    left_inv✝ : Function.LeftInverse g_inv (↑{ toFun := g, map_add' := map_add'✝,  …
    right_inv✝ : Function.RightInverse g_inv (↑{ toFun := g, map_add' := map_add'✝ …
    h : And (Eq f g) (Eq f_inv g_inv)
    ⊢ Eq f g
  -/
  exacts [h.1, h.2]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-30")] alias coe_linearEquiv_injective := toLinearEquiv_injective


theorem coe_injective : @Injective (L₁ ≃ₗ⁅R⁆ L₂) (L₁ → L₂) (↑) :=
  LinearEquiv.coe_injective.comp toLinearEquiv_injective


@[ext]
theorem ext {f g : L₁ ≃ₗ⁅R⁆ L₂} (h : ∀ x, f x = g x) : f = g :=
  coe_injective <| funext h


instance : One (L₁ ≃ₗ⁅R⁆ L₁) :=
  ⟨{ (1 : L₁ ≃ₗ[R] L₁) with map_lie' := rfl }⟩


@[simp]
theorem one_apply (x : L₁) : (1 : L₁ ≃ₗ⁅R⁆ L₁) x = x :=
  rfl


instance : Inhabited (L₁ ≃ₗ⁅R⁆ L₁) :=
  ⟨1⟩


lemma map_lie (e : L₁ ≃ₗ⁅R⁆ L₂) (x y : L₁) : e ⁅x, y⁆ = ⁅e x, e y⁆ :=
  LieHom.map_lie e.toLieHom x y


/-- Lie algebra equivalences are reflexive. -/
def refl : L₁ ≃ₗ⁅R⁆ L₁ :=
  1


@[simp]
theorem refl_apply (x : L₁) : (refl : L₁ ≃ₗ⁅R⁆ L₁) x = x :=
  rfl


/-- Lie algebra equivalences are symmetric. -/
@[symm]
def symm (e : L₁ ≃ₗ⁅R⁆ L₂) : L₂ ≃ₗ⁅R⁆ L₁ :=
  { LieHom.inverse e.toLieHom e.invFun e.left_inv e.right_inv, e.toLinearEquiv.symm with }


@[simp]
theorem symm_symm (e : L₁ ≃ₗ⁅R⁆ L₂) : e.symm.symm = e := rfl


theorem symm_bijective : Function.Bijective (LieEquiv.symm : (L₁ ≃ₗ⁅R⁆ L₂) → L₂ ≃ₗ⁅R⁆ L₁) :=
  Function.bijective_iff_has_inverse.mpr ⟨_, symm_symm, symm_symm⟩


@[simp]
theorem apply_symm_apply (e : L₁ ≃ₗ⁅R⁆ L₂) : ∀ x, e (e.symm x) = x :=
  e.toLinearEquiv.apply_symm_apply


@[simp]
theorem symm_apply_apply (e : L₁ ≃ₗ⁅R⁆ L₂) : ∀ x, e.symm (e x) = x :=
  e.toLinearEquiv.symm_apply_apply


@[simp]
theorem refl_symm : (refl : L₁ ≃ₗ⁅R⁆ L₁).symm = refl :=
  rfl


/-- Lie algebra equivalences are transitive. -/
@[trans]
def trans (e₁ : L₁ ≃ₗ⁅R⁆ L₂) (e₂ : L₂ ≃ₗ⁅R⁆ L₃) : L₁ ≃ₗ⁅R⁆ L₃ :=
  { LieHom.comp e₂.toLieHom e₁.toLieHom, LinearEquiv.trans e₁.toLinearEquiv e₂.toLinearEquiv with }


@[simp]
theorem self_trans_symm (e : L₁ ≃ₗ⁅R⁆ L₂) : e.trans e.symm = refl :=
  ext e.symm_apply_apply


@[simp]
theorem symm_trans_self (e : L₁ ≃ₗ⁅R⁆ L₂) : e.symm.trans e = refl :=
  e.symm.self_trans_symm


@[simp]
theorem trans_apply (e₁ : L₁ ≃ₗ⁅R⁆ L₂) (e₂ : L₂ ≃ₗ⁅R⁆ L₃) (x : L₁) : (e₁.trans e₂) x = e₂ (e₁ x) :=
  rfl


@[simp]
theorem symm_trans (e₁ : L₁ ≃ₗ⁅R⁆ L₂) (e₂ : L₂ ≃ₗ⁅R⁆ L₃) :
    (e₁.trans e₂).symm = e₂.symm.trans e₁.symm :=
  rfl


protected theorem bijective (e : L₁ ≃ₗ⁅R⁆ L₂) : Function.Bijective ((e : L₁ →ₗ⁅R⁆ L₂) : L₁ → L₂) :=
  e.toLinearEquiv.bijective


protected theorem injective (e : L₁ ≃ₗ⁅R⁆ L₂) : Function.Injective ((e : L₁ →ₗ⁅R⁆ L₂) : L₁ → L₂) :=
  e.toLinearEquiv.injective


protected theorem surjective (e : L₁ ≃ₗ⁅R⁆ L₂) :
    Function.Surjective ((e : L₁ →ₗ⁅R⁆ L₂) : L₁ → L₂) :=
  e.toLinearEquiv.surjective


/-- A bijective morphism of Lie algebras yields an equivalence of Lie algebras. -/
@[simps!]
noncomputable def ofBijective (f : L₁ →ₗ⁅R⁆ L₂) (h : Function.Bijective f) : L₁ ≃ₗ⁅R⁆ L₂ :=
  { LinearEquiv.ofBijective (f : L₁ →ₗ[R] L₂)
      h with
    toFun := f
                   /-
                     R : Type u
                     L₁ : Type v
                     L₂ : Type w
                     L₃ : Type w₁
                     inst✝⁶ : CommRing R
                     inst✝⁵ : LieRing L₁
                     inst✝⁴ : LieRing L₂
                     inst✝³ : LieRing L₃
                     inst✝² : LieAlgebra R L₁
                     inst✝¹ : LieAlgebra R L₂
                     inst✝ : LieAlgebra R L₃
                     f : LieHom R L₁ L₂
                     h : Function.Bijective ⇑f
                     ⊢ ∀ {x y : L₁}, Eq ({ toFun := ⇑f, map_add' := ⋯, map_smul' := ⋯ }.toFun (Brac …
                   -/
    map_lie' := by intros x y; exact f.map_lie x y }
                               /-
                                 🎉 no goals
                               -/


/-- A morphism of Lie algebra modules is a linear map which commutes with the action of the Lie
algebra. -/
structure LieModuleHom extends M →ₗ[R] N where
  /-- A module of Lie algebra modules is compatible with the action of the Lie algebra on the
  modules. -/
  map_lie' : ∀ {x : L} {m : M}, toFun ⁅x, m⁆ = ⁅x, toFun m⁆


@[inherit_doc]
notation:25 M " →ₗ⁅" R "," L:25 "⁆ " N:0 => LieModuleHom R L M N


instance : CoeOut (M →ₗ⁅R,L⁆ N) (M →ₗ[R] N) :=
  ⟨LieModuleHom.toLinearMap⟩


instance : FunLike (M →ₗ⁅R, L⁆ N) M N where
  coe f := f.toFun
                             /-
                               R : Type u
                               L : Type v
                               M : Type w
                               N : Type w₁
                               P : Type w₂
                               inst✝¹⁰ : CommRing R
                               inst✝⁹ : LieRing L
                               inst✝⁸ : AddCommGroup M
                               inst✝⁷ : AddCommGroup N
                               inst✝⁶ : AddCommGroup P
                               inst✝⁵ : Module R M
                               inst✝⁴ : Module R N
                               inst✝³ : Module R P
                               inst✝² : LieRingModule L M
                               inst✝¹ : LieRingModule L N
                               inst✝ : LieRingModule L P
                               x y : LieModuleHom R L M N
                               h : Eq ((fun f => (↑f).toFun) x) ((fun f => (↑f).toFun) y)
                               ⊢ Eq x y
                             -/
  coe_injective' x y h := by cases x; cases y; simp at h; simp [h]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp, norm_cast]
theorem coe_toLinearMap (f : M →ₗ⁅R,L⁆ N) : ((f : M →ₗ[R] N) : M → N) = f :=
  rfl


@[simp]
theorem map_smul (f : M →ₗ⁅R,L⁆ N) (c : R) (x : M) : f (c • x) = c • f x :=
  LinearMap.map_smul (f : M →ₗ[R] N) c x


@[simp]
theorem map_add (f : M →ₗ⁅R,L⁆ N) (x y : M) : f (x + y) = f x + f y :=
  LinearMap.map_add (f : M →ₗ[R] N) x y


@[simp]
theorem map_sub (f : M →ₗ⁅R,L⁆ N) (x y : M) : f (x - y) = f x - f y :=
  LinearMap.map_sub (f : M →ₗ[R] N) x y


@[simp]
theorem map_neg (f : M →ₗ⁅R,L⁆ N) (x : M) : f (-x) = -f x :=
  LinearMap.map_neg (f : M →ₗ[R] N) x


@[simp]
theorem map_lie (f : M →ₗ⁅R,L⁆ N) (x : L) (m : M) : f ⁅x, m⁆ = ⁅x, f m⁆ :=
  LieModuleHom.map_lie' f


variable [LieAlgebra R L] [LieModule R L N] [LieModule R L P] in
theorem map_lie₂ (f : M →ₗ⁅R,L⁆ N →ₗ[R] P) (x : L) (m : M) (n : N) :
                                               /-
                                                 R : Type u
                                                 L : Type v
                                                 M : Type w
                                                 N : Type w₁
                                                 P : Type w₂
                                                 inst✝¹³ : CommRing R
                                                 inst✝¹² : LieRing L
                                                 inst✝¹¹ : AddCommGroup M
                                                 inst✝¹⁰ : AddCommGroup N
                                                 inst✝⁹ : AddCommGroup P
                                                 inst✝⁸ : Module R M
                                                 inst✝⁷ : Module R N
                                                 inst✝⁶ : Module R P
                                                 inst✝⁵ : LieRingModule L M
                                                 inst✝⁴ : LieRingModule L N
                                                 inst✝³ : LieRingModule L P
                                                 inst✝² : LieAlgebra R L
                                                 inst✝¹ : LieModule R L N
                                                 inst✝ : LieModule R L P
                                                 f : LieModuleHom R L M (LinearMap (RingHom.id R) N P)
                                                 x : L
                                                 m : M
                                                 n : N
                                                 ⊢ Eq (Bracket.bracket x ((f m) n)) (HAdd.hAdd ((f (Bracket.bracket x m)) n) (( …
                                               -/
    ⁅x, f m n⁆ = f ⁅x, m⁆ n + f m ⁅x, n⁆ := by simp only [sub_add_cancel, map_lie, LieHom.lie_apply]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem map_zero (f : M →ₗ⁅R,L⁆ N) : f 0 = 0 :=
  LinearMap.map_zero (f : M →ₗ[R] N)


/-- The identity map is a morphism of Lie modules. -/
def id : M →ₗ⁅R,L⁆ M :=
  { (LinearMap.id : M →ₗ[R] M) with map_lie' := rfl }


@[simp]
theorem coe_id : ((id : M →ₗ⁅R,L⁆ M) : M → M) = _root_.id :=
  rfl


theorem id_apply (x : M) : (id : M →ₗ⁅R,L⁆ M) x = x :=
  rfl


/-- The constant 0 map is a Lie module morphism. -/
instance : Zero (M →ₗ⁅R,L⁆ N) :=
                                         /-
                                           R : Type u
                                           L : Type v
                                           M : Type w
                                           N : Type w₁
                                           P : Type w₂
                                           inst✝¹⁰ : CommRing R
                                           inst✝⁹ : LieRing L
                                           inst✝⁸ : AddCommGroup M
                                           inst✝⁷ : AddCommGroup N
                                           inst✝⁶ : AddCommGroup P
                                           inst✝⁵ : Module R M
                                           inst✝⁴ : Module R N
                                           inst✝³ : Module R P
                                           inst✝² : LieRingModule L M
                                           inst✝¹ : LieRingModule L N
                                           inst✝ : LieRingModule L P
                                           ⊢ ∀ {x : L} {m : M}, Eq (__src✝.toFun (Bracket.bracket x m)) (Bracket.bracket  …
                                         -/
  ⟨{ (0 : M →ₗ[R] N) with map_lie' := by simp }⟩
                                         /-
                                           🎉 no goals
                                         -/


@[norm_cast, simp]
theorem coe_zero : ⇑(0 : M →ₗ⁅R,L⁆ N) = 0 :=
  rfl


theorem zero_apply (m : M) : (0 : M →ₗ⁅R,L⁆ N) m = 0 :=
  rfl


/-- The identity map is a Lie module morphism. -/
instance : One (M →ₗ⁅R,L⁆ M) :=
  ⟨id⟩


instance : Inhabited (M →ₗ⁅R,L⁆ N) :=
  ⟨0⟩


theorem coe_injective : @Function.Injective (M →ₗ⁅R,L⁆ N) (M → N) (↑) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    N : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : LieRingModule L M
    inst✝ : LieRingModule L N
    ⊢ Function.Injective DFunLike.coe
  -/
  rintro ⟨⟨⟨f, _⟩⟩⟩ ⟨⟨⟨g, _⟩⟩⟩ h
  /-
    case mk.mk.mk.mk.mk.mk
    R : Type u
    L : Type v
    M : Type w
    N : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : LieRingModule L M
    inst✝ : LieRingModule L N
    f : M → N
    map_add'✝¹ : ∀ (x y : M), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    map_smul'✝¹ : ∀ (m : R) (x : M), Eq ({ toFun := f, map_add' := map_add'✝¹ }.to …
    map_lie'✝¹ : ∀ {x : L} {m : M}, Eq ({ toFun := f, map_add' := map_add'✝¹, map_ …
    g : M → N
    map_add'✝ : ∀ (x y : M), Eq (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    map_smul'✝ : ∀ (m : R) (x : M), Eq ({ toFun := g, map_add' := map_add'✝ }.toFu …
    map_lie'✝ : ∀ {x : L} {m : M}, Eq ({ toFun := g, map_add' := map_add'✝, map_sm …
    h : Eq ⇑{ toFun := f, map_add' := map_add'✝¹, map_smul' := map_smul'✝¹, map_li …
    ⊢ Eq { toFun := f, map_add' := map_add'✝¹, map_smul' := map_smul'✝¹, map_lie'  …
  -/
  congr
  /-
    🎉 no goals
  -/


@[ext]
theorem ext {f g : M →ₗ⁅R,L⁆ N} (h : ∀ m, f m = g m) : f = g :=
  coe_injective <| funext h


theorem congr_fun {f g : M →ₗ⁅R,L⁆ N} (h : f = g) (x : M) : f x = g x :=
  h ▸ rfl


@[simp]
theorem mk_coe (f : M →ₗ⁅R,L⁆ N) (h) : (⟨f, h⟩ : M →ₗ⁅R,L⁆ N) = f := by
  /-
    R : Type u
    L : Type v
    M : Type w
    N : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : LieRingModule L M
    inst✝ : LieRingModule L N
    f : LieModuleHom R L M N
    h : ∀ {x : L} {m : M}, Eq ((↑f).toFun (Bracket.bracket x m)) (Bracket.bracket  …
    ⊢ Eq { toLinearMap := ↑f, map_lie' := h } f
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_mk (f : M →ₗ[R] N) (h) : ((⟨f, h⟩ : M →ₗ⁅R,L⁆ N) : M → N) = f := by
  /-
    R : Type u
    L : Type v
    M : Type w
    N : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : LieRingModule L M
    inst✝ : LieRingModule L N
    f : LinearMap (RingHom.id R) M N
    h : ∀ {x : L} {m : M}, Eq (f.toFun (Bracket.bracket x m)) (Bracket.bracket x ( …
    ⊢ Eq ⇑{ toLinearMap := f, map_lie' := h } ⇑f
  -/
  rfl
  /-
    🎉 no goals
  -/


@[norm_cast]
theorem coe_linear_mk (f : M →ₗ[R] N) (h) : ((⟨f, h⟩ : M →ₗ⁅R,L⁆ N) : M →ₗ[R] N) = f := by
  /-
    R : Type u
    L : Type v
    M : Type w
    N : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : LieRingModule L M
    inst✝ : LieRingModule L N
    f : LinearMap (RingHom.id R) M N
    h : ∀ {x : L} {m : M}, Eq (f.toFun (Bracket.bracket x m)) (Bracket.bracket x ( …
    ⊢ Eq (↑{ toLinearMap := f, map_lie' := h }) f
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The composition of Lie module morphisms is a morphism. -/
def comp (f : N →ₗ⁅R,L⁆ P) (g : M →ₗ⁅R,L⁆ N) : M →ₗ⁅R,L⁆ P :=
  { LinearMap.comp f.toLinearMap g.toLinearMap with
    map_lie' := by
      /-
        R : Type u
        L : Type v
        M : Type w
        N : Type w₁
        P : Type w₂
        inst✝¹⁰ : CommRing R
        inst✝⁹ : LieRing L
        inst✝⁸ : AddCommGroup M
        inst✝⁷ : AddCommGroup N
        inst✝⁶ : AddCommGroup P
        inst✝⁵ : Module R M
        inst✝⁴ : Module R N
        inst✝³ : Module R P
        inst✝² : LieRingModule L M
        inst✝¹ : LieRingModule L N
        inst✝ : LieRingModule L P
        f : LieModuleHom R L N P
        g : LieModuleHom R L M N
        ⊢ ∀ {x : L} {m : M}, Eq (__src✝.toFun (Bracket.bracket x m)) (Bracket.bracket  …
      -/
      intros x m
      /-
        R : Type u
        L : Type v
        M : Type w
        N : Type w₁
        P : Type w₂
        inst✝¹⁰ : CommRing R
        inst✝⁹ : LieRing L
        inst✝⁸ : AddCommGroup M
        inst✝⁷ : AddCommGroup N
        inst✝⁶ : AddCommGroup P
        inst✝⁵ : Module R M
        inst✝⁴ : Module R N
        inst✝³ : Module R P
        inst✝² : LieRingModule L M
        inst✝¹ : LieRingModule L N
        inst✝ : LieRingModule L P
        f : LieModuleHom R L N P
        g : LieModuleHom R L M N
        x : L
        m : M
        ⊢ Eq (__src✝.toFun (Bracket.bracket x m)) (Bracket.bracket x (__src✝.toFun m))
      -/
      change f (g ⁅x, m⁆) = ⁅x, f (g m)⁆
      /-
        R : Type u
        L : Type v
        M : Type w
        N : Type w₁
        P : Type w₂
        inst✝¹⁰ : CommRing R
        inst✝⁹ : LieRing L
        inst✝⁸ : AddCommGroup M
        inst✝⁷ : AddCommGroup N
        inst✝⁶ : AddCommGroup P
        inst✝⁵ : Module R M
        inst✝⁴ : Module R N
        inst✝³ : Module R P
        inst✝² : LieRingModule L M
        inst✝¹ : LieRingModule L N
        inst✝ : LieRingModule L P
        f : LieModuleHom R L N P
        g : LieModuleHom R L M N
        x : L
        m : M
        ⊢ Eq (f (g (Bracket.bracket x m))) (Bracket.bracket x (f (g m)))
      -/
      rw [map_lie, map_lie] }
      /-
        🎉 no goals
      -/


theorem comp_apply (f : N →ₗ⁅R,L⁆ P) (g : M →ₗ⁅R,L⁆ N) (m : M) : f.comp g m = f (g m) :=
  rfl


@[norm_cast, simp]
theorem coe_comp (f : N →ₗ⁅R,L⁆ P) (g : M →ₗ⁅R,L⁆ N) : ⇑(f.comp g) = f ∘ g :=
  rfl


@[norm_cast, simp]
theorem toLinearMap_comp (f : N →ₗ⁅R,L⁆ P) (g : M →ₗ⁅R,L⁆ N) :
    (f.comp g : M →ₗ[R] P) = (f : N →ₗ[R] P).comp (g : M →ₗ[R] N) :=
  rfl


/-- The inverse of a bijective morphism of Lie modules is a morphism of Lie modules. -/
def inverse (f : M →ₗ⁅R,L⁆ N) (g : N → M) (h₁ : Function.LeftInverse g f)
    (h₂ : Function.RightInverse g f) : N →ₗ⁅R,L⁆ M :=
  { LinearMap.inverse f.toLinearMap g h₁ h₂ with
    map_lie' := by
      /-
        R : Type u
        L : Type v
        M : Type w
        N : Type w₁
        P : Type w₂
        inst✝¹⁰ : CommRing R
        inst✝⁹ : LieRing L
        inst✝⁸ : AddCommGroup M
        inst✝⁷ : AddCommGroup N
        inst✝⁶ : AddCommGroup P
        inst✝⁵ : Module R M
        inst✝⁴ : Module R N
        inst✝³ : Module R P
        inst✝² : LieRingModule L M
        inst✝¹ : LieRingModule L N
        inst✝ : LieRingModule L P
        f : LieModuleHom R L M N
        g : N → M
        h₁ : Function.LeftInverse g ⇑f
        h₂ : Function.RightInverse g ⇑f
        ⊢ ∀ {x : L} {m : N}, Eq (__src✝.toFun (Bracket.bracket x m)) (Bracket.bracket  …
      -/
      intros x n
      calc
        g ⁅x, n⁆ = g ⁅x, f (g n)⁆ := by rw [h₂]
        _ = g (f ⁅x, g n⁆) := by rw [map_lie]
        _ = ⁅x, g n⁆ := h₁ _
         }


instance : Add (M →ₗ⁅R,L⁆ N) where
                                                                     /-
                                                                       R : Type u
                                                                       L : Type v
                                                                       M : Type w
                                                                       N : Type w₁
                                                                       P : Type w₂
                                                                       inst✝¹⁰ : CommRing R
                                                                       inst✝⁹ : LieRing L
                                                                       inst✝⁸ : AddCommGroup M
                                                                       inst✝⁷ : AddCommGroup N
                                                                       inst✝⁶ : AddCommGroup P
                                                                       inst✝⁵ : Module R M
                                                                       inst✝⁴ : Module R N
                                                                       inst✝³ : Module R P
                                                                       inst✝² : LieRingModule L M
                                                                       inst✝¹ : LieRingModule L N
                                                                       inst✝ : LieRingModule L P
                                                                       f g : LieModuleHom R L M N
                                                                       ⊢ ∀ {x : L} {m : M}, Eq (__src✝.toFun (Bracket.bracket x m)) (Bracket.bracket  …
                                                                     -/
  add f g := { (f : M →ₗ[R] N) + (g : M →ₗ[R] N) with map_lie' := by simp }
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


instance : Sub (M →ₗ⁅R,L⁆ N) where
                                                                     /-
                                                                       R : Type u
                                                                       L : Type v
                                                                       M : Type w
                                                                       N : Type w₁
                                                                       P : Type w₂
                                                                       inst✝¹⁰ : CommRing R
                                                                       inst✝⁹ : LieRing L
                                                                       inst✝⁸ : AddCommGroup M
                                                                       inst✝⁷ : AddCommGroup N
                                                                       inst✝⁶ : AddCommGroup P
                                                                       inst✝⁵ : Module R M
                                                                       inst✝⁴ : Module R N
                                                                       inst✝³ : Module R P
                                                                       inst✝² : LieRingModule L M
                                                                       inst✝¹ : LieRingModule L N
                                                                       inst✝ : LieRingModule L P
                                                                       f g : LieModuleHom R L M N
                                                                       ⊢ ∀ {x : L} {m : M}, Eq (__src✝.toFun (Bracket.bracket x m)) (Bracket.bracket  …
                                                                     -/
  sub f g := { (f : M →ₗ[R] N) - (g : M →ₗ[R] N) with map_lie' := by simp }
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                                                                   /-
                                                                                     R : Type u
                                                                                     L : Type v
                                                                                     M : Type w
                                                                                     N : Type w₁
                                                                                     P : Type w₂
                                                                                     inst✝¹⁰ : CommRing R
                                                                                     inst✝⁹ : LieRing L
                                                                                     inst✝⁸ : AddCommGroup M
                                                                                     inst✝⁷ : AddCommGroup N
                                                                                     inst✝⁶ : AddCommGroup P
                                                                                     inst✝⁵ : Module R M
                                                                                     inst✝⁴ : Module R N
                                                                                     inst✝³ : Module R P
                                                                                     inst✝² : LieRingModule L M
                                                                                     inst✝¹ : LieRingModule L N
                                                                                     inst✝ : LieRingModule L P
                                                                                     f : LieModuleHom R L M N
                                                                                     ⊢ ∀ {x : L} {m : M}, Eq (__src✝.toFun (Bracket.bracket x m)) (Bracket.bracket  …
                                                                                   -/
instance : Neg (M →ₗ⁅R,L⁆ N) where neg f := { -(f : M →ₗ[R] N) with map_lie' := by simp }
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[norm_cast, simp]
theorem coe_add (f g : M →ₗ⁅R,L⁆ N) : ⇑(f + g) = f + g :=
  rfl


theorem add_apply (f g : M →ₗ⁅R,L⁆ N) (m : M) : (f + g) m = f m + g m :=
  rfl


@[norm_cast, simp]
theorem coe_sub (f g : M →ₗ⁅R,L⁆ N) : ⇑(f - g) = f - g :=
  rfl


theorem sub_apply (f g : M →ₗ⁅R,L⁆ N) (m : M) : (f - g) m = f m - g m :=
  rfl


@[norm_cast, simp]
theorem coe_neg (f : M →ₗ⁅R,L⁆ N) : ⇑(-f) = -f :=
  rfl


theorem neg_apply (f : M →ₗ⁅R,L⁆ N) (m : M) : (-f) m = -f m :=
  rfl


instance hasNSMul : SMul ℕ (M →ₗ⁅R,L⁆ N) where
                                                        /-
                                                          R : Type u
                                                          L : Type v
                                                          M : Type w
                                                          N : Type w₁
                                                          P : Type w₂
                                                          inst✝¹⁰ : CommRing R
                                                          inst✝⁹ : LieRing L
                                                          inst✝⁸ : AddCommGroup M
                                                          inst✝⁷ : AddCommGroup N
                                                          inst✝⁶ : AddCommGroup P
                                                          inst✝⁵ : Module R M
                                                          inst✝⁴ : Module R N
                                                          inst✝³ : Module R P
                                                          inst✝² : LieRingModule L M
                                                          inst✝¹ : LieRingModule L N
                                                          inst✝ : LieRingModule L P
                                                          n : Nat
                                                          f : LieModuleHom R L M N
                                                          ⊢ ∀ {x : L} {m : M}, Eq (__src✝.toFun (Bracket.bracket x m)) (Bracket.bracket  …
                                                        -/
  smul n f := { n • (f : M →ₗ[R] N) with map_lie' := by simp }
                                                        /-
                                                          🎉 no goals
                                                        -/


@[norm_cast, simp]
theorem coe_nsmul (n : ℕ) (f : M →ₗ⁅R,L⁆ N) : ⇑(n • f) = n • (⇑f) :=
  rfl


theorem nsmul_apply (n : ℕ) (f : M →ₗ⁅R,L⁆ N) (m : M) : (n • f) m = n • f m :=
  rfl


instance hasZSMul : SMul ℤ (M →ₗ⁅R,L⁆ N) where
                                                        /-
                                                          R : Type u
                                                          L : Type v
                                                          M : Type w
                                                          N : Type w₁
                                                          P : Type w₂
                                                          inst✝¹⁰ : CommRing R
                                                          inst✝⁹ : LieRing L
                                                          inst✝⁸ : AddCommGroup M
                                                          inst✝⁷ : AddCommGroup N
                                                          inst✝⁶ : AddCommGroup P
                                                          inst✝⁵ : Module R M
                                                          inst✝⁴ : Module R N
                                                          inst✝³ : Module R P
                                                          inst✝² : LieRingModule L M
                                                          inst✝¹ : LieRingModule L N
                                                          inst✝ : LieRingModule L P
                                                          z : Int
                                                          f : LieModuleHom R L M N
                                                          ⊢ ∀ {x : L} {m : M}, Eq (__src✝.toFun (Bracket.bracket x m)) (Bracket.bracket  …
                                                        -/
  smul z f := { z • (f : M →ₗ[R] N) with map_lie' := by simp }
                                                        /-
                                                          🎉 no goals
                                                        -/


@[norm_cast, simp]
theorem coe_zsmul (z : ℤ) (f : M →ₗ⁅R,L⁆ N) : ⇑(z • f) = z • (⇑f) :=
  rfl


theorem zsmul_apply (z : ℤ) (f : M →ₗ⁅R,L⁆ N) (m : M) : (z • f) m = z • f m :=
  rfl


instance : AddCommGroup (M →ₗ⁅R,L⁆ N) :=
  coe_injective.addCommGroup _ coe_zero coe_add coe_neg coe_sub (fun _ _ => coe_nsmul _ _)
    (fun _ _ => coe_zsmul _ _)


instance : SMul R (M →ₗ⁅R,L⁆ N) where
                                                        /-
                                                          R : Type u
                                                          L : Type v
                                                          M : Type w
                                                          N : Type w₁
                                                          P : Type w₂
                                                          inst✝¹² : CommRing R
                                                          inst✝¹¹ : LieRing L
                                                          inst✝¹⁰ : AddCommGroup M
                                                          inst✝⁹ : AddCommGroup N
                                                          inst✝⁸ : AddCommGroup P
                                                          inst✝⁷ : Module R M
                                                          inst✝⁶ : Module R N
                                                          inst✝⁵ : Module R P
                                                          inst✝⁴ : LieRingModule L M
                                                          inst✝³ : LieRingModule L N
                                                          inst✝² : LieRingModule L P
                                                          inst✝¹ : LieAlgebra R L
                                                          inst✝ : LieModule R L N
                                                          t : R
                                                          f : LieModuleHom R L M N
                                                          ⊢ ∀ {x : L} {m : M}, Eq (__src✝.toFun (Bracket.bracket x m)) (Bracket.bracket  …
                                                        -/
  smul t f := { t • (f : M →ₗ[R] N) with map_lie' := by simp }
                                                        /-
                                                          🎉 no goals
                                                        -/


@[norm_cast, simp]
theorem coe_smul (t : R) (f : M →ₗ⁅R,L⁆ N) : ⇑(t • f) = t • (⇑f) :=
  rfl


theorem smul_apply (t : R) (f : M →ₗ⁅R,L⁆ N) (m : M) : (t • f) m = t • f m :=
  rfl


instance : Module R (M →ₗ⁅R,L⁆ N) :=
  Function.Injective.module R
    { toFun := fun f => f.toLinearMap.toFun, map_zero' := rfl, map_add' := coe_add }
    coe_injective coe_smul


/-- An equivalence of Lie algebra modules is a linear equivalence which is also a morphism of
Lie algebra modules. -/
structure LieModuleEquiv extends M →ₗ⁅R,L⁆ N where
  /-- The inverse function of an equivalence of Lie modules -/
  invFun : N → M
  /-- The inverse function of an equivalence of Lie modules is a left inverse of the underlying
  function. -/
  left_inv : Function.LeftInverse invFun toFun
  /-- The inverse function of an equivalence of Lie modules is a right inverse of the underlying
  function. -/
  right_inv : Function.RightInverse invFun toFun


@[inherit_doc]
notation:25 M " ≃ₗ⁅" R "," L:25 "⁆ " N:0 => LieModuleEquiv R L M N


/-- View an equivalence of Lie modules as a linear equivalence. -/
def toLinearEquiv (e : M ≃ₗ⁅R,L⁆ N) : M ≃ₗ[R] N :=
  { e with }


/-- View an equivalence of Lie modules as a type level equivalence. -/
def toEquiv (e : M ≃ₗ⁅R,L⁆ N) : M ≃ N :=
  { e with }


instance hasCoeToEquiv : CoeOut (M ≃ₗ⁅R,L⁆ N) (M ≃ N) :=
  ⟨toEquiv⟩


instance hasCoeToLieModuleHom : Coe (M ≃ₗ⁅R,L⁆ N) (M →ₗ⁅R,L⁆ N) :=
  ⟨toLieModuleHom⟩


instance hasCoeToLinearEquiv : CoeOut (M ≃ₗ⁅R,L⁆ N) (M ≃ₗ[R] N) :=
  ⟨toLinearEquiv⟩


instance : EquivLike (M ≃ₗ⁅R,L⁆ N) M N where
  coe f := f.toFun
  inv f := f.invFun
  left_inv f := f.left_inv
  right_inv f := f.right_inv
                                 /-
                                   R : Type u
                                   L : Type v
                                   M : Type w
                                   N : Type w₁
                                   P : Type w₂
                                   inst✝¹⁰ : CommRing R
                                   inst✝⁹ : LieRing L
                                   inst✝⁸ : AddCommGroup M
                                   inst✝⁷ : AddCommGroup N
                                   inst✝⁶ : AddCommGroup P
                                   inst✝⁵ : Module R M
                                   inst✝⁴ : Module R N
                                   inst✝³ : Module R P
                                   inst✝² : LieRingModule L M
                                   inst✝¹ : LieRingModule L N
                                   inst✝ : LieRingModule L P
                                   f g : LieModuleEquiv R L M N
                                   h₁ : Eq ((fun f => (↑f.toLieModuleHom).toFun) f) ((fun f => (↑f.toLieModuleHom …
                                   h₂ : Eq ((fun f => f.invFun) f) ((fun f => f.invFun) g)
                                   ⊢ Eq f g
                                 -/
  coe_injective' f g h₁ h₂ := by cases f; cases g; simp at h₁ h₂; simp [*]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp] lemma coe_coe (e : M ≃ₗ⁅R,L⁆ N) : ⇑(e : M →ₗ⁅R,L⁆ N) = e := rfl


theorem injective (e : M ≃ₗ⁅R,L⁆ N) : Function.Injective e :=
  e.toEquiv.injective


theorem surjective (e : M ≃ₗ⁅R,L⁆ N) : Function.Surjective e :=
  e.toEquiv.surjective


@[simp]
theorem toEquiv_mk (f : M →ₗ⁅R,L⁆ N) (g : N → M) (h₁ h₂) :
    toEquiv (mk f g h₁ h₂ : M ≃ₗ⁅R,L⁆ N) = Equiv.mk f g h₁ h₂ :=
  rfl


@[simp]
theorem coe_mk (f : M →ₗ⁅R,L⁆ N) (invFun h₁ h₂) :
    ((⟨f, invFun, h₁, h₂⟩ : M ≃ₗ⁅R,L⁆ N) : M → N) = f :=
  rfl


theorem coe_toLieModuleHom (e : M ≃ₗ⁅R,L⁆ N) : ⇑(e : M →ₗ⁅R,L⁆ N) = e :=
  rfl


@[deprecated (since := "2024-12-30")] alias coe_to_lieModuleHom := coe_toLieModuleHom


@[simp]
theorem coe_toLinearEquiv (e : M ≃ₗ⁅R,L⁆ N) : ((e : M ≃ₗ[R] N) : M → N) = e :=
  rfl


theorem toEquiv_injective : Function.Injective (toEquiv : (M ≃ₗ⁅R,L⁆ N) → M ≃ N) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    N : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : LieRingModule L M
    inst✝ : LieRingModule L N
    ⊢ Function.Injective LieModuleEquiv.toEquiv
  -/
  rintro ⟨⟨⟨⟨f, -⟩, -⟩, -⟩, f_inv⟩ ⟨⟨⟨⟨g, -⟩, -⟩, -⟩, g_inv⟩
  /-
    case mk.mk.mk.mk.mk.mk.mk.mk
    R : Type u
    L : Type v
    M : Type w
    N : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : LieRingModule L M
    inst✝ : LieRingModule L N
    f_inv : N → M
    f : M → N
    map_add'✝¹ : ∀ (x y : M), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    map_smul'✝¹ : ∀ (m : R) (x : M), Eq ({ toFun := f, map_add' := map_add'✝¹ }.to …
    map_lie'✝¹ : ∀ {x : L} {m : M}, Eq ({ toFun := f, map_add' := map_add'✝¹, map_ …
    left_inv✝¹ : Function.LeftInverse f_inv (↑{ toFun := f, map_add' := map_add'✝¹ …
    right_inv✝¹ : Function.RightInverse f_inv (↑{ toFun := f, map_add' := map_add' …
    g_inv : N → M
    g : M → N
    map_add'✝ : ∀ (x y : M), Eq (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    map_smul'✝ : ∀ (m : R) (x : M), Eq ({ toFun := g, map_add' := map_add'✝ }.toFu …
    map_lie'✝ : ∀ {x : L} {m : M}, Eq ({ toFun := g, map_add' := map_add'✝, map_sm …
    left_inv✝ : Function.LeftInverse g_inv (↑{ toFun := g, map_add' := map_add'✝,  …
    right_inv✝ : Function.RightInverse g_inv (↑{ toFun := g, map_add' := map_add'✝ …
    ⊢ Eq { toFun := f, map_add' := map_add'✝¹, map_smul' := map_smul'✝¹, map_lie'  …
  -/
  intro h
  /-
    case mk.mk.mk.mk.mk.mk.mk.mk
    R : Type u
    L : Type v
    M : Type w
    N : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : LieRingModule L M
    inst✝ : LieRingModule L N
    f_inv : N → M
    f : M → N
    map_add'✝¹ : ∀ (x y : M), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    map_smul'✝¹ : ∀ (m : R) (x : M), Eq ({ toFun := f, map_add' := map_add'✝¹ }.to …
    map_lie'✝¹ : ∀ {x : L} {m : M}, Eq ({ toFun := f, map_add' := map_add'✝¹, map_ …
    left_inv✝¹ : Function.LeftInverse f_inv (↑{ toFun := f, map_add' := map_add'✝¹ …
    right_inv✝¹ : Function.RightInverse f_inv (↑{ toFun := f, map_add' := map_add' …
    g_inv : N → M
    g : M → N
    map_add'✝ : ∀ (x y : M), Eq (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    map_smul'✝ : ∀ (m : R) (x : M), Eq ({ toFun := g, map_add' := map_add'✝ }.toFu …
    map_lie'✝ : ∀ {x : L} {m : M}, Eq ({ toFun := g, map_add' := map_add'✝, map_sm …
    left_inv✝ : Function.LeftInverse g_inv (↑{ toFun := g, map_add' := map_add'✝,  …
    right_inv✝ : Function.RightInverse g_inv (↑{ toFun := g, map_add' := map_add'✝ …
    h : Eq { toFun := f, map_add' := map_add'✝¹, map_smul' := map_smul'✝¹, map_lie …
    ⊢ Eq { toFun := f, map_add' := map_add'✝¹, map_smul' := map_smul'✝¹, map_lie'  …
  -/
  simp only [toEquiv_mk, LieModuleHom.coe_mk, LinearMap.coe_mk, AddHom.coe_mk, Equiv.mk.injEq] at h
  /-
    case mk.mk.mk.mk.mk.mk.mk.mk
    R : Type u
    L : Type v
    M : Type w
    N : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : LieRingModule L M
    inst✝ : LieRingModule L N
    f_inv : N → M
    f : M → N
    map_add'✝¹ : ∀ (x y : M), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    map_smul'✝¹ : ∀ (m : R) (x : M), Eq ({ toFun := f, map_add' := map_add'✝¹ }.to …
    map_lie'✝¹ : ∀ {x : L} {m : M}, Eq ({ toFun := f, map_add' := map_add'✝¹, map_ …
    left_inv✝¹ : Function.LeftInverse f_inv (↑{ toFun := f, map_add' := map_add'✝¹ …
    right_inv✝¹ : Function.RightInverse f_inv (↑{ toFun := f, map_add' := map_add' …
    g_inv : N → M
    g : M → N
    map_add'✝ : ∀ (x y : M), Eq (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    map_smul'✝ : ∀ (m : R) (x : M), Eq ({ toFun := g, map_add' := map_add'✝ }.toFu …
    map_lie'✝ : ∀ {x : L} {m : M}, Eq ({ toFun := g, map_add' := map_add'✝, map_sm …
    left_inv✝ : Function.LeftInverse g_inv (↑{ toFun := g, map_add' := map_add'✝,  …
    right_inv✝ : Function.RightInverse g_inv (↑{ toFun := g, map_add' := map_add'✝ …
    h : And (Eq f g) (Eq f_inv g_inv)
    ⊢ Eq { toFun := f, map_add' := map_add'✝¹, map_smul' := map_smul'✝¹, map_lie'  …
  -/
  congr
  /-
    case mk.mk.mk.mk.mk.mk.mk.mk.e_toLieModuleHom.e_toLinearMap.e_toAddHom.e_toFun
    R : Type u
    L : Type v
    M : Type w
    N : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : LieRingModule L M
    inst✝ : LieRingModule L N
    f_inv : N → M
    f : M → N
    map_add'✝¹ : ∀ (x y : M), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    map_smul'✝¹ : ∀ (m : R) (x : M), Eq ({ toFun := f, map_add' := map_add'✝¹ }.to …
    map_lie'✝¹ : ∀ {x : L} {m : M}, Eq ({ toFun := f, map_add' := map_add'✝¹, map_ …
    left_inv✝¹ : Function.LeftInverse f_inv (↑{ toFun := f, map_add' := map_add'✝¹ …
    right_inv✝¹ : Function.RightInverse f_inv (↑{ toFun := f, map_add' := map_add' …
    g_inv : N → M
    g : M → N
    map_add'✝ : ∀ (x y : M), Eq (g (HAdd.hAdd x y)) (HAdd.hAdd (g x) (g y))
    map_smul'✝ : ∀ (m : R) (x : M), Eq ({ toFun := g, map_add' := map_add'✝ }.toFu …
    map_lie'✝ : ∀ {x : L} {m : M}, Eq ({ toFun := g, map_add' := map_add'✝, map_sm …
    left_inv✝ : Function.LeftInverse g_inv (↑{ toFun := g, map_add' := map_add'✝,  …
    right_inv✝ : Function.RightInverse g_inv (↑{ toFun := g, map_add' := map_add'✝ …
    h : And (Eq f g) (Eq f_inv g_inv)
    ⊢ Eq f g
  -/
  exacts [h.1, h.2]
  /-
    🎉 no goals
  -/


@[ext]
theorem ext (e₁ e₂ : M ≃ₗ⁅R,L⁆ N) (h : ∀ m, e₁ m = e₂ m) : e₁ = e₂ :=
  toEquiv_injective (Equiv.ext h)


instance : One (M ≃ₗ⁅R,L⁆ M) :=
  ⟨{ (1 : M ≃ₗ[R] M) with map_lie' := rfl }⟩


@[simp]
theorem one_apply (m : M) : (1 : M ≃ₗ⁅R,L⁆ M) m = m :=
  rfl


instance : Inhabited (M ≃ₗ⁅R,L⁆ M) :=
  ⟨1⟩


/-- Lie module equivalences are reflexive. -/
@[refl]
def refl : M ≃ₗ⁅R,L⁆ M :=
  1


@[simp]
theorem refl_apply (m : M) : (refl : M ≃ₗ⁅R,L⁆ M) m = m :=
  rfl


/-- Lie module equivalences are symmetric. -/
@[symm]
def symm (e : M ≃ₗ⁅R,L⁆ N) : N ≃ₗ⁅R,L⁆ M :=
  { LieModuleHom.inverse e.toLieModuleHom e.invFun e.left_inv e.right_inv,
    (e : M ≃ₗ[R] N).symm with }


@[simp]
theorem apply_symm_apply (e : M ≃ₗ⁅R,L⁆ N) : ∀ x, e (e.symm x) = x :=
  e.toLinearEquiv.apply_symm_apply


@[simp]
theorem symm_apply_apply (e : M ≃ₗ⁅R,L⁆ N) : ∀ x, e.symm (e x) = x :=
  e.toLinearEquiv.symm_apply_apply


theorem apply_eq_iff_eq_symm_apply {m : M} {n : N} (e : M ≃ₗ⁅R,L⁆ N) :
    e m = n ↔ m = e.symm n :=
  (e : M ≃ N).apply_eq_iff_eq_symm_apply


@[simp]
theorem symm_symm (e : M ≃ₗ⁅R,L⁆ N) : e.symm.symm = e := rfl


theorem symm_bijective :
    Function.Bijective (LieModuleEquiv.symm : (M ≃ₗ⁅R,L⁆ N) → N ≃ₗ⁅R,L⁆ M) :=
  Function.bijective_iff_has_inverse.mpr ⟨_, symm_symm, symm_symm⟩


/-- Lie module equivalences are transitive. -/
@[trans]
def trans (e₁ : M ≃ₗ⁅R,L⁆ N) (e₂ : N ≃ₗ⁅R,L⁆ P) : M ≃ₗ⁅R,L⁆ P :=
  { LieModuleHom.comp e₂.toLieModuleHom e₁.toLieModuleHom,
    LinearEquiv.trans e₁.toLinearEquiv e₂.toLinearEquiv with }


@[simp]
theorem trans_apply (e₁ : M ≃ₗ⁅R,L⁆ N) (e₂ : N ≃ₗ⁅R,L⁆ P) (m : M) : (e₁.trans e₂) m = e₂ (e₁ m) :=
  rfl


@[simp]
theorem symm_trans (e₁ : M ≃ₗ⁅R,L⁆ N) (e₂ : N ≃ₗ⁅R,L⁆ P) :
    (e₁.trans e₂).symm = e₂.symm.trans e₁.symm :=
  rfl


@[simp]
theorem self_trans_symm (e : M ≃ₗ⁅R,L⁆ N) : e.trans e.symm = refl :=
  ext _ _ e.symm_apply_apply


@[simp]
theorem symm_trans_self (e : M ≃ₗ⁅R,L⁆ N) : e.symm.trans e = refl :=
  ext _ _ e.apply_symm_apply


