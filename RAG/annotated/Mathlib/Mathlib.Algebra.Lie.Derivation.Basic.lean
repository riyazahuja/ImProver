/-- A Lie derivation `D` from the Lie `R`-algebra `L` to the `L`-module `M` is an `R`-linear map
that satisfies the Leibniz rule `D [a, b] = [a, D b] - [b, D a]`. -/
structure LieDerivation (R L M : Type*) [CommRing R] [LieRing L] [LieAlgebra R L]
    [AddCommGroup M] [Module R M] [LieRingModule L M] [LieModule R L M]
    extends L →ₗ[R] M where
  protected leibniz' (a b : L) : toLinearMap ⁅a, b⁆ = ⁅a, toLinearMap b⁆ - ⁅b, toLinearMap a⁆


instance : FunLike (LieDerivation R L M) L M where
  coe D := D.toFun
                               /-
                                 R : Type u_1
                                 L : Type u_2
                                 M : Type u_3
                                 inst✝⁶ : CommRing R
                                 inst✝⁵ : LieRing L
                                 inst✝⁴ : LieAlgebra R L
                                 inst✝³ : AddCommGroup M
                                 inst✝² : Module R M
                                 inst✝¹ : LieRingModule L M
                                 inst✝ : LieModule R L M
                                 D D1✝ D2✝ : LieDerivation R L M
                                 a b : L
                                 D1 D2 : LieDerivation R L M
                                 h : Eq ((fun D => D.toFun) D1) ((fun D => D.toFun) D2)
                                 ⊢ Eq D1 D2
                               -/
  coe_injective' D1 D2 h := by cases D1; cases D2; congr; exact DFunLike.coe_injective h
                                                          /-
                                                            🎉 no goals
                                                          -/


instance instLinearMapClass : LinearMapClass (LieDerivation R L M) R L M where
  map_add D := D.toLinearMap.map_add'
  map_smulₛₗ D := D.toLinearMap.map_smul


theorem toFun_eq_coe : D.toFun = ⇑D := rfl


/-- See Note [custom simps projection] -/
def Simps.apply (D : LieDerivation R L M) : L → M := D


instance instCoeToLinearMap : Coe (LieDerivation R L M) (L →ₗ[R] M) :=
  ⟨fun D => D.toLinearMap⟩


@[simp]
theorem mk_coe (f : L →ₗ[R] M) (h₁) : ((⟨f, h₁⟩ : LieDerivation R L M) : L → M) = f :=
  rfl


@[simp, norm_cast]
theorem coeFn_coe (f : LieDerivation R L M) : ⇑(f : L →ₗ[R] M) = f :=
  rfl


theorem coe_injective : @Function.Injective (LieDerivation R L M) (L → M) DFunLike.coe :=
  DFunLike.coe_injective


@[ext]
theorem ext (H : ∀ a, D1 a = D2 a) : D1 = D2 :=
  DFunLike.ext _ _ H


theorem congr_fun (h : D1 = D2) (a : L) : D1 a = D2 a :=
  DFunLike.congr_fun h a


@[simp]
lemma apply_lie_eq_sub (D : LieDerivation R L M) (a b : L) :
    D ⁅a, b⁆ = ⁅a, D b⁆ - ⁅b, D a⁆ :=
  D.leibniz' a b


/-- For a Lie derivation from a Lie algebra to itself, the usual Leibniz rule holds. -/
lemma apply_lie_eq_add (D : LieDerivation R L L) (a b : L) :
    D ⁅a, b⁆ = ⁅a, D b⁆ + ⁅D a, b⁆ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    D : LieDerivation R L L
    a b : L
    ⊢ Eq (D (Bracket.bracket a b)) (HAdd.hAdd (Bracket.bracket a (D b)) (Bracket.b …
  -/
  rw [LieDerivation.apply_lie_eq_sub, sub_eq_add_neg, lie_skew]
  /-
    🎉 no goals
  -/


/-- Two Lie derivations equal on a set are equal on its Lie span. -/
theorem eqOn_lieSpan {s : Set L} (h : Set.EqOn D1 D2 s) :
    Set.EqOn D1 D2 (LieSubalgebra.lieSpan R L s) :=
    fun _ hz =>
                                    /-
                                      R : Type u_1
                                      L : Type u_2
                                      M : Type u_3
                                      inst✝⁶ : CommRing R
                                      inst✝⁵ : LieRing L
                                      inst✝⁴ : LieAlgebra R L
                                      inst✝³ : AddCommGroup M
                                      inst✝² : Module R M
                                      inst✝¹ : LieRingModule L M
                                      inst✝ : LieModule R L M
                                      D1 D2 : LieDerivation R L M
                                      s : Set L
                                      h : Set.EqOn (⇑D1) (⇑D2) s
                                      x✝ : L
                                      hz : Membership.mem (↑(LieSubalgebra.lieSpan R L s)) x✝
                                      ⊢ Eq (D1 0) (D2 0)
                                    -/
      have zero : D1 0 = D2 0 := by simp only [map_zero]
                                    /-
                                      🎉 no goals
                                    -/
      have smul : ∀ (r : R), ∀ {x : L}, D1 x = D2 x → D1 (r • x) = D2 (r • x) :=
                         /-
                           R : Type u_1
                           L : Type u_2
                           M : Type u_3
                           inst✝⁶ : CommRing R
                           inst✝⁵ : LieRing L
                           inst✝⁴ : LieAlgebra R L
                           inst✝³ : AddCommGroup M
                           inst✝² : Module R M
                           inst✝¹ : LieRingModule L M
                           inst✝ : LieModule R L M
                           D1 D2 : LieDerivation R L M
                           s : Set L
                           h : Set.EqOn (⇑D1) (⇑D2) s
                           x✝² : L
                           hz : Membership.mem (↑(LieSubalgebra.lieSpan R L s)) x✝²
                           zero : Eq (D1 0) (D2 0)
                           x✝¹ : R
                           x✝ : L
                           hx : Eq (D1 x✝) (D2 x✝)
                           ⊢ Eq (D1 (HSMul.hSMul x✝¹ x✝)) (D2 (HSMul.hSMul x✝¹ x✝))
                         -/
        fun _ _ hx => by simp only [map_smul, hx]
                         /-
                           🎉 no goals
                         -/
      have add : ∀ x y, D1 x = D2 x → D1 y = D2 y → D1 (x + y) = D2 (x + y) :=
                            /-
                              R : Type u_1
                              L : Type u_2
                              M : Type u_3
                              inst✝⁶ : CommRing R
                              inst✝⁵ : LieRing L
                              inst✝⁴ : LieAlgebra R L
                              inst✝³ : AddCommGroup M
                              inst✝² : Module R M
                              inst✝¹ : LieRingModule L M
                              inst✝ : LieModule R L M
                              D1 D2 : LieDerivation R L M
                              s : Set L
                              h : Set.EqOn (⇑D1) (⇑D2) s
                              x✝² : L
                              hz : Membership.mem (↑(LieSubalgebra.lieSpan R L s)) x✝²
                              zero : Eq (D1 0) (D2 0)
                              smul : ∀ (r : R) {x : L}, Eq (D1 x) (D2 x) → Eq (D1 (HSMul.hSMul r x)) (D2 (HS …
                              x✝¹ x✝ : L
                              hx : Eq (D1 x✝¹) (D2 x✝¹)
                              hy : Eq (D1 x✝) (D2 x✝)
                              ⊢ Eq (D1 (HAdd.hAdd x✝¹ x✝)) (D2 (HAdd.hAdd x✝¹ x✝))
                            -/
        fun _ _ hx hy => by simp only [map_add, hx, hy]
                            /-
                              🎉 no goals
                            -/
      have lie : ∀ x y, D1 x = D2 x → D1 y = D2 y → D1 ⁅x, y⁆ = D2 ⁅x, y⁆ :=
                            /-
                              R : Type u_1
                              L : Type u_2
                              M : Type u_3
                              inst✝⁶ : CommRing R
                              inst✝⁵ : LieRing L
                              inst✝⁴ : LieAlgebra R L
                              inst✝³ : AddCommGroup M
                              inst✝² : Module R M
                              inst✝¹ : LieRingModule L M
                              inst✝ : LieModule R L M
                              D1 D2 : LieDerivation R L M
                              s : Set L
                              h : Set.EqOn (⇑D1) (⇑D2) s
                              x✝² : L
                              hz : Membership.mem (↑(LieSubalgebra.lieSpan R L s)) x✝²
                              zero : Eq (D1 0) (D2 0)
                              smul : ∀ (r : R) {x : L}, Eq (D1 x) (D2 x) → Eq (D1 (HSMul.hSMul r x)) (D2 (HS …
                              add : ∀ (x y : L), Eq (D1 x) (D2 x) → Eq (D1 y) (D2 y) → Eq (D1 (HAdd.hAdd x y …
                              x✝¹ x✝ : L
                              hx : Eq (D1 x✝¹) (D2 x✝¹)
                              hy : Eq (D1 x✝) (D2 x✝)
                              ⊢ Eq (D1 (Bracket.bracket x✝¹ x✝)) (D2 (Bracket.bracket x✝¹ x✝))
                            -/
        fun _ _ hx hy => by simp only [apply_lie_eq_sub, hx, hy]
                            /-
                              🎉 no goals
                            -/
      LieSubalgebra.lieSpan_induction R (p := fun x => D1 x = D2 x) hz h zero smul add lie


/-- If the Lie span of a set is the whole Lie algebra, then two Lie derivations equal on this set
are equal on the whole Lie algebra. -/
theorem ext_of_lieSpan_eq_top (s : Set L) (hs : LieSubalgebra.lieSpan R L s = ⊤)
    (h : Set.EqOn D1 D2 s) : D1 = D2 :=
  ext fun _ => eqOn_lieSpan h <| hs.symm ▸ trivial


/-- The general Leibniz rule for Lie derivatives. -/
theorem iterate_apply_lie (D : LieDerivation R L L) (n : ℕ) (a b : L) :
    D^[n] ⁅a, b⁆ = ∑ ij in antidiagonal n, choose n ij.1 • ⁅D^[ij.1] a, D^[ij.2] b⁆ := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [sum_antidiagonal_choose_succ_nsmul (M := L) (fun i j => ⁅D^[i] a, D^[j] b⁆) n]
    simp only [Function.iterate_succ_apply', ih, map_sum, map_nsmul, apply_lie_eq_add, smul_add,
      sum_add_distrib, add_right_inj]
    refine sum_congr rfl fun ⟨i, j⟩ hij ↦ ?_
    rw [n.choose_symm_of_eq_add (mem_antidiagonal.1 hij).symm]


/-- Alternate version of the general Leibniz rule for Lie derivatives. -/
theorem iterate_apply_lie' (D : LieDerivation R L L) (n : ℕ) (a b : L) :
    D^[n] ⁅a, b⁆ = ∑ i in range (n + 1), n.choose i • ⁅D^[i] a, D^[n - i] b⁆ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    D : LieDerivation R L L
    n : Nat
    a b : L
    ⊢ Eq (Nat.iterate (⇑D) n (Bracket.bracket a b)) ((Finset.range (HAdd.hAdd n 1) …
  -/
  rw [iterate_apply_lie D n a b]
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    D : LieDerivation R L L
    n : Nat
    a b : L
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal n).sum fun ij => HSMul.hSMul (n.cho …
  -/
  exact sum_antidiagonal_eq_sum_range_succ (fun i j ↦ n.choose i • ⁅D^[i] a, D^[j] b⁆) n
  /-
    🎉 no goals
  -/


instance instZero : Zero (LieDerivation R L M) where
  zero :=
    { toLinearMap := 0
                                /-
                                  R : Type u_1
                                  L : Type u_2
                                  M : Type u_3
                                  inst✝⁶ : CommRing R
                                  inst✝⁵ : LieRing L
                                  inst✝⁴ : LieAlgebra R L
                                  inst✝³ : AddCommGroup M
                                  inst✝² : Module R M
                                  inst✝¹ : LieRingModule L M
                                  inst✝ : LieModule R L M
                                  D D1 D2 : LieDerivation R L M
                                  a✝ b✝ a b : L
                                  ⊢ Eq (0 (Bracket.bracket a b)) (HSub.hSub (Bracket.bracket a (0 b)) (Bracket.b …
                                -/
      leibniz' := fun a b => by simp only [LinearMap.zero_apply, lie_zero, sub_self] }
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem coe_zero : ⇑(0 : LieDerivation R L M) = 0 :=
  rfl


@[simp]
theorem coe_zero_linearMap : ↑(0 : LieDerivation R L M) = (0 : L →ₗ[R] M) :=
  rfl


theorem zero_apply (a : L) : (0 : LieDerivation R L M) a = 0 :=
  rfl


instance : Inhabited (LieDerivation R L M) :=
  ⟨0⟩


instance instAdd : Add (LieDerivation R L M) where
  add D1 D2 :=
    { toLinearMap := D1 + D2
      leibniz' := fun a b ↦ by
        /-
          R : Type u_1
          L : Type u_2
          M : Type u_3
          inst✝⁶ : CommRing R
          inst✝⁵ : LieRing L
          inst✝⁴ : LieAlgebra R L
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : LieRingModule L M
          inst✝ : LieModule R L M
          D D1✝ D2✝ : LieDerivation R L M
          a✝ b✝ : L
          D1 D2 : LieDerivation R L M
          a b : L
          ⊢ Eq ((HAdd.hAdd ↑D1 ↑D2) (Bracket.bracket a b)) (HSub.hSub (Bracket.bracket a …
        -/
        simp only [LinearMap.add_apply, coeFn_coe, apply_lie_eq_sub, lie_add, add_sub_add_comm] }
        /-
          🎉 no goals
        -/


@[simp]
theorem coe_add (D1 D2 : LieDerivation R L M) : ⇑(D1 + D2) = D1 + D2 :=
  rfl


@[simp]
theorem coe_add_linearMap (D1 D2 : LieDerivation R L M) : ↑(D1 + D2) = (D1 + D2 : L →ₗ[R] M) :=
  rfl


theorem add_apply : (D1 + D2) a = D1 a + D2 a :=
  rfl


protected theorem map_neg : D (-a) = -D a :=
  map_neg D a


protected theorem map_sub : D (a - b) = D a - D b :=
  map_sub D a b


instance instNeg : Neg (LieDerivation R L M) :=
  ⟨fun D =>
    mk (-D) fun a b => by
      simp only [LinearMap.neg_apply, coeFn_coe, apply_lie_eq_sub,
        neg_sub, lie_neg, sub_neg_eq_add, add_comm, ← sub_eq_add_neg] ⟩


@[simp]
theorem coe_neg (D : LieDerivation R L M) : ⇑(-D) = -D :=
  rfl


@[simp]
theorem coe_neg_linearMap (D : LieDerivation R L M) : ↑(-D) = (-D : L →ₗ[R] M) :=
  rfl


theorem neg_apply : (-D) a = -D a :=
  rfl


instance instSub : Sub (LieDerivation R L M) :=
  ⟨fun D1 D2 =>
    mk (D1 - D2 : L →ₗ[R] M) fun a b => by
      /-
        R : Type u_1
        L : Type u_2
        M : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra R L
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        D D1✝ D2✝ : LieDerivation R L M
        a✝ b✝ : L
        D1 D2 : LieDerivation R L M
        a b : L
        ⊢ Eq ((HSub.hSub ↑D1 ↑D2) (Bracket.bracket a b)) (HSub.hSub (Bracket.bracket a …
      -/
      simp only [LinearMap.sub_apply, coeFn_coe, apply_lie_eq_sub, lie_sub, sub_sub_sub_comm]⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_sub (D1 D2 : LieDerivation R L M) : ⇑(D1 - D2) = D1 - D2 :=
  rfl


@[simp]
theorem coe_sub_linearMap (D1 D2 : LieDerivation R L M) : ↑(D1 - D2) = (D1 - D2 : L →ₗ[R] M) :=
  rfl


theorem sub_apply {D1 D2 : LieDerivation R L M} : (D1 - D2) a = D1 a - D2 a :=
  rfl


/-- A typeclass mixin saying that scalar multiplication and Lie bracket are left commutative. -/
class SMulBracketCommClass (S L α : Type*) [SMul S α] [LieRing L] [AddCommGroup α]
    [LieRingModule L α] : Prop where
  /-- `•` and `⁅⬝, ⬝⁆`  are left commutative -/
  smul_bracket_comm : ∀ (s : S) (l : L) (a : α), s • ⁅l, a⁆ = ⁅l, s • a⁆


instance instSMul : SMul S (LieDerivation R L M) where
  smul r D :=
    { toLinearMap := r • D
      leibniz' := fun a b => by simp only [LinearMap.smul_apply, coeFn_coe, apply_lie_eq_sub,
        smul_sub, SMulBracketCommClass.smul_bracket_comm] }


@[simp]
theorem coe_smul (r : S) (D : LieDerivation R L M) : ⇑(r • D) = r • ⇑D :=
  rfl


@[simp]
theorem coe_smul_linearMap (r : S) (D : LieDerivation R L M) : ↑(r • D) = r • (D : L →ₗ[R] M) :=
  rfl


theorem smul_apply (r : S) (D : LieDerivation R L M) : (r • D) a = r • D a :=
  rfl


instance instSMulBase : SMulBracketCommClass R L M := ⟨fun s l a ↦ (lie_smul s l a).symm⟩


instance instSMulNat : SMulBracketCommClass ℕ L M := ⟨fun s l a => (lie_nsmul l a s).symm⟩


instance instSMulInt : SMulBracketCommClass ℤ L M := ⟨fun s l a => (lie_zsmul l a s).symm⟩


instance instAddCommGroup : AddCommGroup (LieDerivation R L M) :=
  coe_injective.addCommGroup _ coe_zero coe_add coe_neg coe_sub (fun _ _ => rfl) fun _ _ => rfl


/-- `coe_fn` as an `AddMonoidHom`. -/
def coeFnAddMonoidHom : LieDerivation R L M →+ L → M where
  toFun := (↑)
  map_zero' := coe_zero
  map_add' := coe_add


instance : DistribMulAction S (LieDerivation R L M) :=
  Function.Injective.distribMulAction coeFnAddMonoidHom coe_injective coe_smul


instance [SMul S T] [IsScalarTower S T M] : IsScalarTower S T (LieDerivation R L M) :=
  ⟨fun _ _ _ => ext fun _ => smul_assoc _ _ _⟩


instance [SMulCommClass S T M] : SMulCommClass S T (LieDerivation R L M) :=
  ⟨fun _ _ _ => ext fun _ => smul_comm _ _ _⟩


instance instModule {S : Type*} [Semiring S] [Module S M] [SMulCommClass R S M]
    [SMulBracketCommClass S L M] : Module S (LieDerivation R L M) :=
  Function.Injective.module S coeFnAddMonoidHom coe_injective coe_smul


/-- The commutator of two Lie derivations on a Lie algebra is a Lie derivation. -/
instance instBracket : Bracket (LieDerivation R L L) (LieDerivation R L L) where
  bracket D1 D2 := LieDerivation.mk ⁅(D1 : Module.End R L), (D2 : Module.End R L)⁆ (fun a b => by
    simp only [Ring.lie_def, apply_lie_eq_add, coeFn_coe,
      LinearMap.sub_apply, LinearMap.mul_apply, map_add, sub_lie, lie_sub, ← lie_skew b]
    /-
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      D1 D2 : LieDerivation R L L
      a b : L
      ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Bracket.bracket a (D1 (D2 b))) (Bracket …
    -/
    /-
      🎉 no goals
    -/
    abel)
    /-
      🎉 no goals
    -/


@[simp]
lemma commutator_coe_linear_map : ↑⁅D1, D2⁆ = ⁅(D1 : Module.End R L), (D2 : Module.End R L)⁆ :=
  rfl


lemma commutator_apply (a : L) : ⁅D1, D2⁆ a = D1 (D2 a) - D2 (D1 a) :=
  rfl


instance : LieRing (LieDerivation R L L) where
  add_lie d e f := by
    /-
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      D1 D2 d e f : LieDerivation R L L
      ⊢ Eq (Bracket.bracket (HAdd.hAdd d e) f) (HAdd.hAdd (Bracket.bracket d f) (Bra …
    -/
                                                             /-
                                                               🎉 no goals
                                                             -/
    ext a; simp only [commutator_apply, add_apply, map_add]; abel
                                                             /-
                                                               🎉 no goals
                                                             -/
  lie_add d e f := by
    /-
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      D1 D2 d e f : LieDerivation R L L
      ⊢ Eq (Bracket.bracket d (HAdd.hAdd e f)) (HAdd.hAdd (Bracket.bracket d e) (Bra …
    -/
                                                             /-
                                                               🎉 no goals
                                                             -/
    ext a; simp only [commutator_apply, add_apply, map_add]; abel
                                                             /-
                                                               🎉 no goals
                                                             -/
  lie_self d := by
    /-
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      D1 D2 d : LieDerivation R L L
      ⊢ Eq (Bracket.bracket d d) 0
    -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
    ext a; simp only [commutator_apply, add_apply, map_add, zero_apply]; abel
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  leibniz_lie d e f := by
    /-
      R : Type u_1
      L : Type u_2
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      D1 D2 d e f : LieDerivation R L L
      ⊢ Eq (Bracket.bracket d (Bracket.bracket e f)) (HAdd.hAdd (Bracket.bracket (Br …
    -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    ext a; simp only [commutator_apply, add_apply, sub_apply, map_sub]; abel
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- The set of Lie derivations from a Lie algebra `L` to itself is a Lie algebra. -/
instance instLieAlgebra : LieAlgebra R (LieDerivation R L L) where
                              /-
                                R : Type u_1
                                L : Type u_2
                                inst✝² : CommRing R
                                inst✝¹ : LieRing L
                                inst✝ : LieAlgebra R L
                                D1 D2 : LieDerivation R L L
                                r : R
                                d e : LieDerivation R L L
                                ⊢ Eq (Bracket.bracket d (HSMul.hSMul r e)) (HSMul.hSMul r (Bracket.bracket d e))
                              -/
  lie_smul := fun r d e => by ext a; simp only [commutator_apply, map_smul, smul_sub, smul_apply]
                                     /-
                                       🎉 no goals
                                     -/


@[simp] lemma lie_apply (D₁ D₂ : LieDerivation R L L) (x : L) :
    ⁅D₁, D₂⁆ x = D₁ (D₂ x) - D₂ (D₁ x) :=
  rfl


/-- The Lie algebra morphism from Lie derivations into linear endormophisms. -/
def toLinearMapLieHom : LieDerivation R L L →ₗ⁅R⁆ L →ₗ[R] L where
  toFun := toLinearMap
                 /-
                   R : Type u_1
                   L : Type u_2
                   inst✝² : CommRing R
                   inst✝¹ : LieRing L
                   inst✝ : LieAlgebra R L
                   ⊢ ∀ (x y : LieDerivation R L L), Eq (↑(HAdd.hAdd x y)) (HAdd.hAdd ↑x ↑y)
                 -/
  map_add' := by intro D1 D2; dsimp
                              /-
                                🎉 no goals
                              -/
                  /-
                    R : Type u_1
                    L : Type u_2
                    inst✝² : CommRing R
                    inst✝¹ : LieRing L
                    inst✝ : LieAlgebra R L
                    ⊢ ∀ (m : R) (x : LieDerivation R L L), Eq ({ toFun := LieDerivation.toLinearMa …
                  -/
  map_smul' := by intro D1 D2; dsimp
                               /-
                                 🎉 no goals
                               -/
                 /-
                   R : Type u_1
                   L : Type u_2
                   inst✝² : CommRing R
                   inst✝¹ : LieRing L
                   inst✝ : LieAlgebra R L
                   ⊢ ∀ {x y : LieDerivation R L L}, Eq ({ toFun := LieDerivation.toLinearMap, map …
                 -/
  map_lie' := by intro D1 D2; dsimp
                              /-
                                🎉 no goals
                              -/


/-- The map from Lie derivations to linear endormophisms is injective. -/
lemma toLinearMapLieHom_injective : Function.Injective (toLinearMapLieHom R L) :=
  fun _ _ h ↦ ext fun a ↦ congrFun (congrArg DFunLike.coe h) a


/-- Lie derivations over a Noetherian Lie algebra form a Noetherian module. -/
instance instNoetherian [IsNoetherian R L] : IsNoetherian R (LieDerivation R L L) :=
  isNoetherian_of_linearEquiv (LinearEquiv.ofInjective _ (toLinearMapLieHom_injective R L)).symm


/-- The natural map from a Lie module to the derivations taking values in it. -/
@[simps!]
def inner : M →ₗ[R] LieDerivation R L M where
  toFun m :=
    { __ := (LieModule.toEnd R L M : L →ₗ[R] Module.End R M).flip m
                     /-
                       R : Type u_1
                       L : Type u_2
                       M : Type u_3
                       inst✝⁶ : CommRing R
                       inst✝⁵ : LieRing L
                       inst✝⁴ : LieAlgebra R L
                       inst✝³ : AddCommGroup M
                       inst✝² : Module R M
                       inst✝¹ : LieRingModule L M
                       inst✝ : LieModule R L M
                       m : M
                       ⊢ ∀ (a b : L), Eq (__spread✝⁻⁰ (Bracket.bracket a b)) (HSub.hSub (Bracket.brac …
                     -/
      leibniz' := by simp }
                     /-
                       🎉 no goals
                     -/
                     /-
                       R : Type u_1
                       L : Type u_2
                       M : Type u_3
                       inst✝⁶ : CommRing R
                       inst✝⁵ : LieRing L
                       inst✝⁴ : LieAlgebra R L
                       inst✝³ : AddCommGroup M
                       inst✝² : Module R M
                       inst✝¹ : LieRingModule L M
                       inst✝ : LieModule R L M
                       m n : M
                       ⊢ Eq
                           ((fun m =>
                               let __spread.0 := (↑(LieModule.toEnd R L M)).flip m;
                               { toLinearMap := __spread.0, leibniz' := ⋯ })
                             (HAdd.hAdd m n))
                           (HAdd.hAdd
                             ((fun m =>
                                 let __spread.0 := (↑(LieModule.toEnd R L M)).flip m;
                                 { toLinearMap := __spread.0, leibniz' := ⋯ })
                               m)
                             ((fun m =>
                                 let __spread.0 := (↑(LieModule.toEnd R L M)).flip m;
                                 { toLinearMap := __spread.0, leibniz' := ⋯ })
                               n))
                     -/
  map_add' m n := by ext; simp
                          /-
                            🎉 no goals
                          -/
                      /-
                        R : Type u_1
                        L : Type u_2
                        M : Type u_3
                        inst✝⁶ : CommRing R
                        inst✝⁵ : LieRing L
                        inst✝⁴ : LieAlgebra R L
                        inst✝³ : AddCommGroup M
                        inst✝² : Module R M
                        inst✝¹ : LieRingModule L M
                        inst✝ : LieModule R L M
                        t : R
                        m : M
                        ⊢ Eq
                            ({
                                  toFun := fun m =>
                                    let __spread.0 := (↑(LieModule.toEnd R L M)).flip m;
                                    { toLinearMap := __spread.0, leibniz' := ⋯ },
                                  map_add' := ⋯ }.toFun
                              (HSMul.hSMul t m))
                            (HSMul.hSMul ((RingHom.id R) t)
                              ({
                                    toFun := fun m =>
                                      let __spread.0 := (↑(LieModule.toEnd R L M)).flip m;
                                      { toLinearMap := __spread.0, leibniz' := ⋯ },
                                    map_add' := ⋯ }.toFun
                                m))
                      -/
  map_smul' t m := by ext; simp
                           /-
                             🎉 no goals
                           -/


instance instLieRingModule : LieRingModule L (LieDerivation R L M) where
  bracket x D := inner R L M (D x)
                      /-
                        R : Type u_1
                        L : Type u_2
                        M : Type u_3
                        inst✝⁶ : CommRing R
                        inst✝⁵ : LieRing L
                        inst✝⁴ : LieAlgebra R L
                        inst✝³ : AddCommGroup M
                        inst✝² : Module R M
                        inst✝¹ : LieRingModule L M
                        inst✝ : LieModule R L M
                        x y : L
                        D : LieDerivation R L M
                        ⊢ Eq (Bracket.bracket (HAdd.hAdd x y) D) (HAdd.hAdd (Bracket.bracket x D) (Bra …
                      -/
  add_lie x y D := by simp
                      /-
                        🎉 no goals
                      -/
                        /-
                          R : Type u_1
                          L : Type u_2
                          M : Type u_3
                          inst✝⁶ : CommRing R
                          inst✝⁵ : LieRing L
                          inst✝⁴ : LieAlgebra R L
                          inst✝³ : AddCommGroup M
                          inst✝² : Module R M
                          inst✝¹ : LieRingModule L M
                          inst✝ : LieModule R L M
                          x : L
                          D₁ D₂ : LieDerivation R L M
                          ⊢ Eq (Bracket.bracket x (HAdd.hAdd D₁ D₂)) (HAdd.hAdd (Bracket.bracket x D₁) ( …
                        -/
  lie_add x D₁ D₂ := by simp
                        /-
                          🎉 no goals
                        -/
                          /-
                            R : Type u_1
                            L : Type u_2
                            M : Type u_3
                            inst✝⁶ : CommRing R
                            inst✝⁵ : LieRing L
                            inst✝⁴ : LieAlgebra R L
                            inst✝³ : AddCommGroup M
                            inst✝² : Module R M
                            inst✝¹ : LieRingModule L M
                            inst✝ : LieModule R L M
                            x y : L
                            D : LieDerivation R L M
                            ⊢ Eq (Bracket.bracket x (Bracket.bracket y D)) (HAdd.hAdd (Bracket.bracket (Br …
                          -/
  leibniz_lie x y D := by simp
                          /-
                            🎉 no goals
                          -/


@[simp] lemma lie_lieDerivation_apply (x y : L) (D : LieDerivation R L M) :
    ⁅x, D⁆ y = ⁅y, D x⁆ :=
  rfl


@[simp] lemma lie_coe_lieDerivation_apply (x : L) (D : LieDerivation R L M) :
    ⁅x, (D : L →ₗ[R] M)⁆ = ⁅x, D⁆ := by
  /-
    R : Type u_1
    L : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    x : L
    D : LieDerivation R L M
    ⊢ Eq (Bracket.bracket x ↑D) ↑(Bracket.bracket x D)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


instance instLieModule : LieModule R L (LieDerivation R L M) where
                       /-
                         R : Type u_1
                         L : Type u_2
                         M : Type u_3
                         inst✝⁶ : CommRing R
                         inst✝⁵ : LieRing L
                         inst✝⁴ : LieAlgebra R L
                         inst✝³ : AddCommGroup M
                         inst✝² : Module R M
                         inst✝¹ : LieRingModule L M
                         inst✝ : LieModule R L M
                         t : R
                         x : L
                         D : LieDerivation R L M
                         ⊢ Eq (Bracket.bracket (HSMul.hSMul t x) D) (HSMul.hSMul t (Bracket.bracket x D))
                       -/
  smul_lie t x D := by ext; simp
                            /-
                              🎉 no goals
                            -/
                       /-
                         R : Type u_1
                         L : Type u_2
                         M : Type u_3
                         inst✝⁶ : CommRing R
                         inst✝⁵ : LieRing L
                         inst✝⁴ : LieAlgebra R L
                         inst✝³ : AddCommGroup M
                         inst✝² : Module R M
                         inst✝¹ : LieRingModule L M
                         inst✝ : LieModule R L M
                         t : R
                         x : L
                         D : LieDerivation R L M
                         ⊢ Eq (Bracket.bracket x (HSMul.hSMul t D)) (HSMul.hSMul t (Bracket.bracket x D))
                       -/
  lie_smul t x D := by ext; simp
                            /-
                              🎉 no goals
                            -/


protected lemma leibniz_lie (x : L) (D₁ D₂ : LieDerivation R L L) :
    ⁅x, ⁅D₁, D₂⁆⁆ = ⁅⁅x, D₁⁆, D₂⁆ + ⁅D₁, ⁅x, D₂⁆⁆ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x : L
    D₁ D₂ : LieDerivation R L L
    ⊢ Eq (Bracket.bracket x (Bracket.bracket D₁ D₂)) (HAdd.hAdd (Bracket.bracket ( …
  -/
  ext y
  /-
    case H
    R : Type u_1
    L : Type u_2
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    x : L
    D₁ D₂ : LieDerivation R L L
    y : L
    ⊢ Eq ((Bracket.bracket x (Bracket.bracket D₁ D₂)) y) ((HAdd.hAdd (Bracket.brac …
  -/
  simp [-lie_skew, ← lie_skew (D₁ x) (D₂ y), ← lie_skew (D₂ x) (D₁ y), sub_eq_neg_add]
  /-
    🎉 no goals
  -/


