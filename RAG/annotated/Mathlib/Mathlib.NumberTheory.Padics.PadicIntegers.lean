/-- The `p`-adic integers `ℤ_[p]` are the `p`-adic numbers with norm `≤ 1`. -/
def PadicInt : Type := {x : ℚ_[p] // ‖x‖ ≤ 1}


/-- The ring of `p`-adic integers. -/
notation "ℤ_[" p "]" => PadicInt p


instance : Coe ℤ_[p] ℚ_[p] :=
  ⟨Subtype.val⟩


theorem ext {x y : ℤ_[p]} : (x : ℚ_[p]) = y → x = y :=
  Subtype.ext


/-- The `p`-adic integers as a subring of `ℚ_[p]`. -/
def subring : Subring ℚ_[p] where
  carrier := { x : ℚ_[p] | ‖x‖ ≤ 1 }
                  /-
                    p : Nat
                    hp : Fact (Nat.Prime p)
                    x y : PadicInt p
                    ⊢ Membership.mem { carrier := setOf fun x => LE.le (Norm.norm x) 1, mul_mem' : …
                  -/
                 /-
                   p : Nat
                   hp : Fact (Nat.Prime p)
                   x y : PadicInt p
                   ⊢ Membership.mem { carrier := setOf fun x => LE.le (Norm.norm x) 1, mul_mem' : …
                 -/
  zero_mem' := by norm_num
                 /-
                   🎉 no goals
                 -/
                  /-
                    🎉 no goals
                  -/
  one_mem' := by norm_num
  add_mem' hx hy := (padicNormE.nonarchimedean _ _).trans <| max_le_iff.2 ⟨hx, hy⟩
  mul_mem' hx hy := (padicNormE.mul _ _).trans_le <| mul_le_one₀ hx (norm_nonneg _) hy
  neg_mem' hx := (norm_neg _).trans_le hx


@[simp]
theorem mem_subring_iff {x : ℚ_[p]} : x ∈ subring p ↔ ‖x‖ ≤ 1 := Iff.rfl


/-- Addition on `ℤ_[p]` is inherited from `ℚ_[p]`. -/
                            /-
                              p : Nat
                              hp : Fact (Nat.Prime p)
                              x y : PadicInt p
                              ⊢ Add (Subtype fun x => Membership.mem (PadicInt.subring p) x)
                            -/
instance : Add ℤ_[p] := (by infer_instance : Add (subring p))
                            /-
                              🎉 no goals
                            -/


/-- Multiplication on `ℤ_[p]` is inherited from `ℚ_[p]`. -/
                            /-
                              p : Nat
                              hp : Fact (Nat.Prime p)
                              x y : PadicInt p
                              ⊢ Mul (Subtype fun x => Membership.mem (PadicInt.subring p) x)
                            -/
instance : Mul ℤ_[p] := (by infer_instance : Mul (subring p))
                            /-
                              🎉 no goals
                            -/


/-- Negation on `ℤ_[p]` is inherited from `ℚ_[p]`. -/
                            /-
                              p : Nat
                              hp : Fact (Nat.Prime p)
                              x y : PadicInt p
                              ⊢ Neg (Subtype fun x => Membership.mem (PadicInt.subring p) x)
                            -/
instance : Neg ℤ_[p] := (by infer_instance : Neg (subring p))
                            /-
                              🎉 no goals
                            -/


/-- Subtraction on `ℤ_[p]` is inherited from `ℚ_[p]`. -/
                            /-
                              p : Nat
                              hp : Fact (Nat.Prime p)
                              x y : PadicInt p
                              ⊢ Sub (Subtype fun x => Membership.mem (PadicInt.subring p) x)
                            -/
instance : Sub ℤ_[p] := (by infer_instance : Sub (subring p))
                            /-
                              🎉 no goals
                            -/


/-- Zero on `ℤ_[p]` is inherited from `ℚ_[p]`. -/
                             /-
                               p : Nat
                               hp : Fact (Nat.Prime p)
                               x y : PadicInt p
                               ⊢ Zero (Subtype fun x => Membership.mem (PadicInt.subring p) x)
                             -/
instance : Zero ℤ_[p] := (by infer_instance : Zero (subring p))
                             /-
                               🎉 no goals
                             -/


instance : Inhabited ℤ_[p] := ⟨0⟩


/-- One on `ℤ_[p]` is inherited from `ℚ_[p]`. -/
                                /-
                                  p : Nat
                                  hp : Fact (Nat.Prime p)
                                  x y : PadicInt p
                                  ⊢ LE.le (Norm.norm 1) 1
                                -/
instance : One ℤ_[p] := ⟨⟨1, by norm_num⟩⟩
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem mk_zero {h} : (⟨0, h⟩ : ℤ_[p]) = (0 : ℤ_[p]) := rfl


@[simp, norm_cast]
theorem coe_add (z1 z2 : ℤ_[p]) : ((z1 + z2 : ℤ_[p]) : ℚ_[p]) = z1 + z2 := rfl


@[simp, norm_cast]
theorem coe_mul (z1 z2 : ℤ_[p]) : ((z1 * z2 : ℤ_[p]) : ℚ_[p]) = z1 * z2 := rfl


@[simp, norm_cast]
theorem coe_neg (z1 : ℤ_[p]) : ((-z1 : ℤ_[p]) : ℚ_[p]) = -z1 := rfl


@[simp, norm_cast]
theorem coe_sub (z1 z2 : ℤ_[p]) : ((z1 - z2 : ℤ_[p]) : ℚ_[p]) = z1 - z2 := rfl


@[simp, norm_cast]
theorem coe_one : ((1 : ℤ_[p]) : ℚ_[p]) = 1 := rfl


@[simp, norm_cast]
theorem coe_zero : ((0 : ℤ_[p]) : ℚ_[p]) = 0 := rfl


                                                          /-
                                                            p : Nat
                                                            hp : Fact (Nat.Prime p)
                                                            x : PadicInt p
                                                            ⊢ Iff (Eq (↑x) 0) (Eq x 0)
                                                          -/
@[simp] lemma coe_eq_zero : (x : ℚ_[p]) = 0 ↔ x = 0 := by rw [← coe_zero, Subtype.coe_inj]
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma coe_ne_zero : (x : ℚ_[p]) ≠ 0 ↔ x ≠ 0 := coe_eq_zero.not


                                     /-
                                       p : Nat
                                       hp : Fact (Nat.Prime p)
                                       x y : PadicInt p
                                       ⊢ AddCommGroup (Subtype fun x => Membership.mem (PadicInt.subring p) x)
                                     -/
instance : AddCommGroup ℤ_[p] := (by infer_instance : AddCommGroup (subring p))
                                     /-
                                       🎉 no goals
                                     -/


                                              /-
                                                p : Nat
                                                hp : Fact (Nat.Prime p)
                                                x y : PadicInt p
                                                ⊢ CommRing (Subtype fun x => Membership.mem (PadicInt.subring p) x)
                                              -/
instance instCommRing : CommRing ℤ_[p] := (by infer_instance : CommRing (subring p))
                                              /-
                                                🎉 no goals
                                              -/


@[simp, norm_cast]
theorem coe_natCast (n : ℕ) : ((n : ℤ_[p]) : ℚ_[p]) = n := rfl


@[deprecated (since := "2024-04-17")]
alias coe_nat_cast := coe_natCast


@[simp, norm_cast]
theorem coe_intCast (z : ℤ) : ((z : ℤ_[p]) : ℚ_[p]) = z := rfl


@[deprecated (since := "2024-04-17")]
alias coe_int_cast := coe_intCast


/-- The coercion from `ℤ_[p]` to `ℚ_[p]` as a ring homomorphism. -/
def Coe.ringHom : ℤ_[p] →+* ℚ_[p] := (subring p).subtype


@[simp, norm_cast]
theorem coe_pow (x : ℤ_[p]) (n : ℕ) : (↑(x ^ n) : ℚ_[p]) = (↑x : ℚ_[p]) ^ n := rfl


                                                          /-
                                                            p : Nat
                                                            hp : Fact (Nat.Prime p)
                                                            k : PadicInt p
                                                            ⊢ Eq ⟨↑k, ⋯⟩ k
                                                          -/
theorem mk_coe (k : ℤ_[p]) : (⟨k, k.2⟩ : ℤ_[p]) = k := by simp
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- The inverse of a `p`-adic integer with norm equal to `1` is also a `p`-adic integer.
Otherwise, the inverse is defined to be `0`. -/
def inv : ℤ_[p] → ℤ_[p]
                                           /-
                                             p : Nat
                                             hp : Fact (Nat.Prime p)
                                             x y : PadicInt p
                                             k : Padic p
                                             property✝ : LE.le (Norm.norm k) 1
                                             h : Eq (Norm.norm k) 1
                                             ⊢ LE.le (Norm.norm (Inv.inv k)) 1
                                           -/
  | ⟨k, _⟩ => if h : ‖k‖ = 1 then ⟨k⁻¹, by simp [h]⟩ else 0
                                           /-
                                             🎉 no goals
                                           -/


instance : CharZero ℤ_[p] where
  cast_injective m n h :=
                                        /-
                                          p : Nat
                                          hp : Fact (Nat.Prime p)
                                          x y : PadicInt p
                                          m n : Nat
                                          h : Eq ↑m ↑n
                                          ⊢ Eq ↑m ↑n
                                        -/
    Nat.cast_injective (R := ℚ_[p]) (by rw [Subtype.ext_iff] at h; norm_cast at h)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[norm_cast]
                                                                   /-
                                                                     p : Nat
                                                                     hp : Fact (Nat.Prime p)
                                                                     z1 z2 : Int
                                                                     ⊢ Iff (Eq ↑z1 ↑z2) (Eq z1 z2)
                                                                   -/
theorem intCast_eq (z1 z2 : ℤ) : (z1 : ℤ_[p]) = z2 ↔ z1 = z2 := by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[deprecated (since := "2024-04-05")] alias coe_int_eq := intCast_eq


/-- A sequence of integers that is Cauchy with respect to the `p`-adic norm converges to a `p`-adic
integer. -/
def ofIntSeq (seq : ℕ → ℤ) (h : IsCauSeq (padicNorm p) fun n => seq n) : ℤ_[p] :=
  ⟨⟦⟨_, h⟩⟧,
    show ↑(PadicSeq.norm _) ≤ (1 : ℝ) by
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        x y : PadicInt p
        seq : Nat → Int
        h : IsCauSeq (padicNorm p) fun n => ↑(seq n)
        ⊢ LE.le (↑(PadicSeq.norm ⟨fun n => ↑(seq n), h⟩)) 1
      -/
      rw [PadicSeq.norm]
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        x y : PadicInt p
        seq : Nat → Int
        h : IsCauSeq (padicNorm p) fun n => ↑(seq n)
        ⊢ LE.le (↑(dite (HasEquiv.Equiv ⟨fun n => ↑(seq n), h⟩ 0) (fun hf => 0) fun hf …
      -/
                             /-
                               🎉 no goals
                             -/
      split_ifs with hne <;> norm_cast
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        x y : PadicInt p
        seq : Nat → Int
        h : IsCauSeq (padicNorm p) fun n => ↑(seq n)
        hne : Not (HasEquiv.Equiv ⟨fun n => ↑(seq n), h⟩ 0)
        ⊢ LE.le (padicNorm p (↑⟨fun n => ↑(seq n), h⟩ (PadicSeq.stationaryPoint hne))) 1
      -/
      apply padicNorm.of_int⟩
      /-
        🎉 no goals
      -/


instance : MetricSpace ℤ_[p] := Subtype.metricSpace


instance : IsUltrametricDist ℤ_[p] := IsUltrametricDist.subtype _


instance completeSpace : CompleteSpace ℤ_[p] :=
  have : IsClosed { x : ℚ_[p] | ‖x‖ ≤ 1 } := isClosed_le continuous_norm continuous_const
  this.completeSpace_coe


instance : Norm ℤ_[p] := ⟨fun z => ‖(z : ℚ_[p])‖⟩


theorem norm_def {z : ℤ_[p]} : ‖z‖ = ‖(z : ℚ_[p])‖ := rfl


instance : NormedCommRing ℤ_[p] :=
  { PadicInt.instCommRing with
    dist_eq := fun ⟨_, _⟩ ⟨_, _⟩ => rfl
                   /-
                     p : Nat
                     hp : Fact (Nat.Prime p)
                     x y : PadicInt p
                     ⊢ ∀ (a b : PadicInt p), LE.le (Norm.norm (HMul.hMul a b)) (HMul.hMul (Norm.nor …
                   -/
    norm_mul := by simp [norm_def]
                   /-
                     🎉 no goals
                   -/
    norm := norm }


instance : NormOneClass ℤ_[p] :=
  ⟨norm_def.trans norm_one⟩


instance isAbsoluteValue : IsAbsoluteValue fun z : ℤ_[p] => ‖z‖ where
  abv_nonneg' := norm_nonneg
                     /-
                       p : Nat
                       hp : Fact (Nat.Prime p)
                       x y : PadicInt p
                       ⊢ ∀ {x : PadicInt p}, Iff (Eq (Norm.norm x) 0) (Eq x 0)
                     -/
  abv_eq_zero' := by simp [norm_eq_zero]
                     /-
                       🎉 no goals
                     -/
  abv_add' := fun ⟨_, _⟩ ⟨_, _⟩ => norm_add_le _ _
                     /-
                       p : Nat
                       hp : Fact (Nat.Prime p)
                       x y x✝¹ x✝ : PadicInt p
                       ⊢ Eq (Norm.norm (HMul.hMul x✝¹ x✝)) (HMul.hMul (Norm.norm x✝¹) (Norm.norm x✝))
                     -/
  abv_mul' _ _ := by simp only [norm_def, padicNormE.mul, PadicInt.coe_mul]
                     /-
                       🎉 no goals
                     -/


instance : IsDomain ℤ_[p] := Function.Injective.isDomain (subring p).subtype Subtype.coe_injective


theorem norm_le_one (z : ℤ_[p]) : ‖z‖ ≤ 1 := z.2


@[simp]
                                                                 /-
                                                                   p : Nat
                                                                   hp : Fact (Nat.Prime p)
                                                                   z1 z2 : PadicInt p
                                                                   ⊢ Eq (Norm.norm (HMul.hMul z1 z2)) (HMul.hMul (Norm.norm z1) (Norm.norm z2))
                                                                 -/
theorem norm_mul (z1 z2 : ℤ_[p]) : ‖z1 * z2‖ = ‖z1‖ * ‖z2‖ := by simp [norm_def]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem norm_pow (z : ℤ_[p]) : ∀ n : ℕ, ‖z ^ n‖ = ‖z‖ ^ n
            /-
              p : Nat
              hp : Fact (Nat.Prime p)
              z : PadicInt p
              ⊢ Eq (Norm.norm (HPow.hPow z 0)) (HPow.hPow (Norm.norm z) 0)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | k + 1 => by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      z : PadicInt p
      k : Nat
      ⊢ Eq (Norm.norm (HPow.hPow z (HAdd.hAdd k 1))) (HPow.hPow (Norm.norm z) (HAdd. …
    -/
    rw [pow_succ, pow_succ, norm_mul]
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      z : PadicInt p
      k : Nat
      ⊢ Eq (HMul.hMul (Norm.norm (HPow.hPow z k)) (Norm.norm z)) (HMul.hMul (HPow.hP …
    -/
    congr
    /-
      case e_a
      p : Nat
      hp : Fact (Nat.Prime p)
      z : PadicInt p
      k : Nat
      ⊢ Eq (Norm.norm (HPow.hPow z k)) (HPow.hPow (Norm.norm z) k)
    -/
    apply norm_pow
    /-
      🎉 no goals
    -/


theorem nonarchimedean (q r : ℤ_[p]) : ‖q + r‖ ≤ max ‖q‖ ‖r‖ := padicNormE.nonarchimedean _ _


theorem norm_add_eq_max_of_ne {q r : ℤ_[p]} : ‖q‖ ≠ ‖r‖ → ‖q + r‖ = max ‖q‖ ‖r‖ :=
  padicNormE.add_eq_max_of_ne


theorem norm_eq_of_norm_add_lt_right {z1 z2 : ℤ_[p]} (h : ‖z1 + z2‖ < ‖z2‖) : ‖z1‖ = ‖z2‖ :=
  by_contra fun hne =>
                     /-
                       p : Nat
                       hp : Fact (Nat.Prime p)
                       z1 z2 : PadicInt p
                       h : LT.lt (Norm.norm (HAdd.hAdd z1 z2)) (Norm.norm z2)
                       hne : Not (Eq (Norm.norm z1) (Norm.norm z2))
                       ⊢ GE.ge (Norm.norm (HAdd.hAdd z1 z2)) (Norm.norm z2)
                     -/
    not_lt_of_ge (by rw [norm_add_eq_max_of_ne hne]; apply le_max_right) h
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem norm_eq_of_norm_add_lt_left {z1 z2 : ℤ_[p]} (h : ‖z1 + z2‖ < ‖z1‖) : ‖z1‖ = ‖z2‖ :=
  by_contra fun hne =>
                     /-
                       p : Nat
                       hp : Fact (Nat.Prime p)
                       z1 z2 : PadicInt p
                       h : LT.lt (Norm.norm (HAdd.hAdd z1 z2)) (Norm.norm z1)
                       hne : Not (Eq (Norm.norm z1) (Norm.norm z2))
                       ⊢ GE.ge (Norm.norm (HAdd.hAdd z1 z2)) (Norm.norm z1)
                     -/
    not_lt_of_ge (by rw [norm_add_eq_max_of_ne hne]; apply le_max_left) h
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
                                                                         /-
                                                                           p : Nat
                                                                           hp : Fact (Nat.Prime p)
                                                                           z : PadicInt p
                                                                           ⊢ Eq (Norm.norm ↑z) (Norm.norm z)
                                                                         -/
theorem padic_norm_e_of_padicInt (z : ℤ_[p]) : ‖(z : ℚ_[p])‖ = ‖z‖ := by simp [norm_def]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


                                                                                 /-
                                                                                   p : Nat
                                                                                   hp : Fact (Nat.Prime p)
                                                                                   z : Int
                                                                                   ⊢ Eq (Norm.norm ↑z) (Norm.norm ↑z)
                                                                                 -/
theorem norm_intCast_eq_padic_norm (z : ℤ) : ‖(z : ℤ_[p])‖ = ‖(z : ℚ_[p])‖ := by simp [norm_def]
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


@[deprecated (since := "2024-04-17")]
alias norm_int_cast_eq_padic_norm := norm_intCast_eq_padic_norm


@[simp]
theorem norm_eq_padic_norm {q : ℚ_[p]} (hq : ‖q‖ ≤ 1) : @norm ℤ_[p] _ ⟨q, hq⟩ = ‖q‖ := rfl


@[simp]
theorem norm_p : ‖(p : ℤ_[p])‖ = (p : ℝ)⁻¹ := padicNormE.norm_p


                                                                          /-
                                                                            p : Nat
                                                                            hp : Fact (Nat.Prime p)
                                                                            n : Nat
                                                                            ⊢ Eq (Norm.norm (HPow.hPow (↑p) n)) (HPow.hPow (↑p) (Neg.neg ↑n))
                                                                          -/
theorem norm_p_pow (n : ℕ) : ‖(p : ℤ_[p]) ^ n‖ = (p : ℝ) ^ (-n : ℤ) := by simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


private def cauSeq_to_rat_cauSeq (f : CauSeq ℤ_[p] norm) : CauSeq ℚ_[p] fun a => ‖a‖ :=
                                /-
                                  p : Nat
                                  hp : Fact (Nat.Prime p)
                                  x y : PadicInt p
                                  f : CauSeq (PadicInt p) Norm.norm
                                  x✝ : Real
                                  hε : GT.gt x✝ 0
                                  ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt ((fun a => Norm.norm a) (HSub …
                                -/
  ⟨fun n => f n, fun _ hε => by simpa [norm, norm_def] using f.cauchy hε⟩
                                /-
                                  🎉 no goals
                                -/


instance complete : CauSeq.IsComplete ℤ_[p] norm :=
  ⟨fun f =>
    have hqn : ‖CauSeq.lim (cauSeq_to_rat_cauSeq f)‖ ≤ 1 :=
      padicNormE_lim_le zero_lt_one fun _ => norm_le_one _
    ⟨⟨_, hqn⟩, fun ε => by
      /-
        p : Nat
        hp : Fact (Nat.Prime p)
        x y : PadicInt p
        f : CauSeq (PadicInt p) Norm.norm
        hqn : LE.le (Norm.norm (PadicInt.cauSeq_to_rat_cauSeq f).lim) 1
        ε : Real
        ⊢ GT.gt ε 0 → Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (Norm.norm (↑(HSu …
      -/
      simpa [norm, norm_def] using CauSeq.equiv_lim (cauSeq_to_rat_cauSeq f) ε⟩⟩
      /-
        🎉 no goals
      -/


theorem exists_pow_neg_lt {ε : ℝ} (hε : 0 < ε) : ∃ k : ℕ, (p : ℝ) ^ (-(k : ℤ)) < ε := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun k => LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
  -/
  obtain ⟨k, hk⟩ := exists_nat_gt ε⁻¹
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    ε : Real
    hε : LT.lt 0 ε
    k : Nat
    hk : LT.lt (Inv.inv ε) ↑k
    ⊢ Exists fun k => LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
  -/
  use k
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    ε : Real
    hε : LT.lt 0 ε
    k : Nat
    hk : LT.lt (Inv.inv ε) ↑k
    ⊢ LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
  -/
  rw [← inv_lt_inv₀ hε (zpow_pos _ _)]
    /-
      case h
      p : Nat
      hp : Fact (Nat.Prime p)
      ε : Real
      hε : LT.lt 0 ε
      k : Nat
      hk : LT.lt (Inv.inv ε) ↑k
      ⊢ LT.lt (Inv.inv ε) (Inv.inv (HPow.hPow (↑p) (Neg.neg ↑k)))
    -/
  · rw [zpow_neg, inv_inv, zpow_natCast]
    /-
      case h
      p : Nat
      hp : Fact (Nat.Prime p)
      ε : Real
      hε : LT.lt 0 ε
      k : Nat
      hk : LT.lt (Inv.inv ε) ↑k
      ⊢ LT.lt (Inv.inv ε) (HPow.hPow (↑p) k)
    -/
    apply lt_of_lt_of_le hk
    /-
      case h
      p : Nat
      hp : Fact (Nat.Prime p)
      ε : Real
      hε : LT.lt 0 ε
      k : Nat
      hk : LT.lt (Inv.inv ε) ↑k
      ⊢ LE.le (↑k) (HPow.hPow (↑p) k)
    -/
    norm_cast
    /-
      case h
      p : Nat
      hp : Fact (Nat.Prime p)
      ε : Real
      hε : LT.lt 0 ε
      k : Nat
      hk : LT.lt (Inv.inv ε) ↑k
      ⊢ LE.le k (HPow.hPow p k)
    -/
    apply le_of_lt
    /-
      case h.hab
      p : Nat
      hp : Fact (Nat.Prime p)
      ε : Real
      hε : LT.lt 0 ε
      k : Nat
      hk : LT.lt (Inv.inv ε) ↑k
      ⊢ LT.lt k (HPow.hPow p k)
    -/
    convert Nat.lt_pow_self _ using 1
    /-
      case h.hab.convert_3
      p : Nat
      hp : Fact (Nat.Prime p)
      ε : Real
      hε : LT.lt 0 ε
      k : Nat
      hk : LT.lt (Inv.inv ε) ↑k
      ⊢ LT.lt 1 p
    -/
    exact hp.1.one_lt
    /-
      🎉 no goals
    -/
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      ε : Real
      hε : LT.lt 0 ε
      k : Nat
      hk : LT.lt (Inv.inv ε) ↑k
      ⊢ LT.lt 0 ↑p
    -/
  · exact mod_cast hp.1.pos
    /-
      🎉 no goals
    -/


theorem exists_pow_neg_lt_rat {ε : ℚ} (hε : 0 < ε) : ∃ k : ℕ, (p : ℚ) ^ (-(k : ℤ)) < ε := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ε : Rat
    hε : LT.lt 0 ε
    ⊢ Exists fun k => LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
  -/
  obtain ⟨k, hk⟩ := @exists_pow_neg_lt p _ ε (mod_cast hε)
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    ε : Rat
    hε : LT.lt 0 ε
    k : Nat
    hk : LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ↑ε
    ⊢ Exists fun k => LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
  -/
  use k
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    ε : Rat
    hε : LT.lt 0 ε
    k : Nat
    hk : LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ↑ε
    ⊢ LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
  -/
  rw [show (p : ℝ) = (p : ℚ) by simp] at hk
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    ε : Rat
    hε : LT.lt 0 ε
    k : Nat
    hk : LT.lt (HPow.hPow (↑↑p) (Neg.neg ↑k)) ↑ε
    ⊢ LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
  -/
  exact mod_cast hk
  /-
    🎉 no goals
  -/


theorem norm_int_lt_one_iff_dvd (k : ℤ) : ‖(k : ℤ_[p])‖ < 1 ↔ (p : ℤ) ∣ k :=
                                         /-
                                           p : Nat
                                           hp : Fact (Nat.Prime p)
                                           k : Int
                                           this : Iff (LT.lt (Norm.norm ↑k) 1) (Dvd.dvd (↑p) k)
                                           ⊢ Iff (LT.lt (Norm.norm ↑k) 1) (Dvd.dvd (↑p) k)
                                         -/
  suffices ‖(k : ℚ_[p])‖ < 1 ↔ ↑p ∣ k by rwa [norm_intCast_eq_padic_norm]
                                         /-
                                           🎉 no goals
                                         -/
  padicNormE.norm_int_lt_one_iff_dvd k


theorem norm_int_le_pow_iff_dvd {k : ℤ} {n : ℕ} :
    ‖(k : ℤ_[p])‖ ≤ (p : ℝ) ^ (-n : ℤ) ↔ (p ^ n : ℤ) ∣ k :=
  suffices ‖(k : ℚ_[p])‖ ≤ (p : ℝ) ^ (-n : ℤ) ↔ (p ^ n : ℤ) ∣ k by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Int
      n : Nat
      this : Iff (LE.le (Norm.norm ↑k) (HPow.hPow (↑p) (Neg.neg ↑n))) (Dvd.dvd (HPow …
      ⊢ Iff (LE.le (Norm.norm ↑k) (HPow.hPow (↑p) (Neg.neg ↑n))) (Dvd.dvd (HPow.hPow …
    -/
    simpa [norm_intCast_eq_padic_norm]
    /-
      🎉 no goals
    -/
  padicNormE.norm_int_le_pow_iff_dvd _ _


lemma valuation_coe_nonneg : 0 ≤ (x : ℚ_[p]).valuation := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ LE.le 0 (↑x).valuation
  -/
  obtain rfl | hx := eq_or_ne x 0
    /-
      case inl
      p : Nat
      hp : Fact (Nat.Prime p)
      ⊢ LE.le 0 (↑0).valuation
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    hx : Ne x 0
    ⊢ LE.le 0 (↑x).valuation
  -/
  have := x.2
  rwa [Padic.norm_eq_zpow_neg_valuation <| coe_ne_zero.2 hx, zpow_le_one_iff_right₀, neg_nonpos]
    at this
  /-
    case inr
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    hx : Ne x 0
    this : LE.le (HPow.hPow (↑p) (Neg.neg (↑x).valuation)) 1
    ⊢ LT.lt 1 ↑p
  -/
  exact mod_cast hp.out.one_lt
  /-
    🎉 no goals
  -/


/-- `PadicInt.valuation` lifts the `p`-adic valuation on `ℚ` to `ℤ_[p]`. -/
def valuation (x : ℤ_[p]) : ℕ := (x : ℚ_[p]).valuation.toNat


@[simp, norm_cast] lemma valuation_coe (x : ℤ_[p]) : (x : ℚ_[p]).valuation = x.valuation := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ Eq (↑x).valuation ↑x.valuation
  -/
  simp [valuation, valuation_coe_nonneg]
  /-
    🎉 no goals
  -/


                                                               /-
                                                                 p : Nat
                                                                 hp : Fact (Nat.Prime p)
                                                                 ⊢ Eq (PadicInt.valuation 0) 0
                                                               -/
@[simp] lemma valuation_zero : valuation (0 : ℤ_[p]) = 0 := by simp [valuation]
                                                               /-
                                                                 🎉 no goals
                                                               -/

                                                              /-
                                                                p : Nat
                                                                hp : Fact (Nat.Prime p)
                                                                ⊢ Eq (PadicInt.valuation 1) 0
                                                              -/
@[simp] lemma valuation_one : valuation (1 : ℤ_[p]) = 0 := by simp [valuation]
                                                              /-
                                                                🎉 no goals
                                                              -/

                                                            /-
                                                              p : Nat
                                                              hp : Fact (Nat.Prime p)
                                                              ⊢ Eq (↑p).valuation 1
                                                            -/
@[simp] lemma valuation_p : valuation (p : ℤ_[p]) = 1 := by simp [valuation]
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma le_valuation_add (hxy : x + y ≠ 0) : min x.valuation y.valuation ≤ (x + y).valuation := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x y : PadicInt p
    hxy : Ne (HAdd.hAdd x y) 0
    ⊢ LE.le (Min.min x.valuation y.valuation) (HAdd.hAdd x y).valuation
  -/
  zify; simpa [← valuation_coe] using Padic.le_valuation_add <| coe_ne_zero.2 hxy
        /-
          🎉 no goals
        -/


@[simp] lemma valuation_mul (hx : x ≠ 0) (hy : y ≠ 0) :
    (x * y).valuation = x.valuation + y.valuation := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x y : PadicInt p
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (HMul.hMul x y).valuation (HAdd.hAdd x.valuation y.valuation)
  -/
  zify; simp [← valuation_coe, Padic.valuation_mul (coe_ne_zero.2 hx) (coe_ne_zero.2 hy)]
        /-
          🎉 no goals
        -/


@[simp]
lemma valuation_pow (x : ℤ_[p]) (n : ℕ) : (x ^ n).valuation = n * x.valuation := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    n : Nat
    ⊢ Eq (HPow.hPow x n).valuation (HMul.hMul n x.valuation)
  -/
  zify; simp [← valuation_coe]
        /-
          🎉 no goals
        -/


lemma norm_eq_zpow_neg_valuation {x : ℤ_[p]} (hx : x ≠ 0) : ‖x‖ = p ^ (-x.valuation : ℤ) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    hx : Ne x 0
    ⊢ Eq (Norm.norm x) (HPow.hPow (↑p) (Neg.neg ↑x.valuation))
  -/
  simp [norm_def, Padic.norm_eq_zpow_neg_valuation <| coe_ne_zero.2 hx]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-10")] alias norm_eq_pow_val := norm_eq_zpow_neg_valuation

-- TODO: Do we really need this lemma?

@[simp]
theorem valuation_p_pow_mul (n : ℕ) (c : ℤ_[p]) (hc : c ≠ 0) :
    ((p : ℤ_[p]) ^ n * c).valuation = n + c.valuation := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    c : PadicInt p
    hc : Ne c 0
    ⊢ Eq (HMul.hMul (HPow.hPow (↑p) n) c).valuation (HAdd.hAdd n c.valuation)
  -/
  rw [valuation_mul (NeZero.ne _) hc, valuation_pow, valuation_p, mul_one]
  /-
    🎉 no goals
  -/


theorem mul_inv : ∀ {z : ℤ_[p]}, ‖z‖ = 1 → z * z.inv = 1
  | ⟨k, _⟩, h => by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Padic p
      property✝ : LE.le (Norm.norm k) 1
      h : Eq (Norm.norm ⟨k, property✝⟩) 1
      ⊢ Eq (HMul.hMul ⟨k, property✝⟩ (PadicInt.inv ⟨k, property✝⟩)) 1
    -/
    have hk : k ≠ 0 := fun h' => zero_ne_one' ℚ_[p] (by simp [h'] at h)
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Padic p
      property✝ : LE.le (Norm.norm k) 1
      h : Eq (Norm.norm ⟨k, property✝⟩) 1
      hk : Ne k 0
      ⊢ Eq (HMul.hMul ⟨k, property✝⟩ (PadicInt.inv ⟨k, property✝⟩)) 1
    -/
    unfold PadicInt.inv
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Padic p
      property✝ : LE.le (Norm.norm k) 1
      h : Eq (Norm.norm ⟨k, property✝⟩) 1
      hk : Ne k 0
      ⊢ Eq (HMul.hMul ⟨k, property✝⟩ (PadicInt.inv.match_1 (fun x => PadicInt p) ⟨k, …
    -/
    rw [norm_eq_padic_norm] at h
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Padic p
      property✝ : LE.le (Norm.norm k) 1
      h : Eq (Norm.norm k) 1
      hk : Ne k 0
      ⊢ Eq (HMul.hMul ⟨k, property✝⟩ (PadicInt.inv.match_1 (fun x => PadicInt p) ⟨k, …
    -/
    dsimp only
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Padic p
      property✝ : LE.le (Norm.norm k) 1
      h : Eq (Norm.norm k) 1
      hk : Ne k 0
      ⊢ Eq (HMul.hMul ⟨k, property✝⟩ (dite (Eq (Norm.norm k) 1) (fun h => ⟨Inv.inv k …
    -/
    rw [dif_pos h]
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Padic p
      property✝ : LE.le (Norm.norm k) 1
      h : Eq (Norm.norm k) 1
      hk : Ne k 0
      ⊢ Eq (HMul.hMul ⟨k, property✝⟩ ⟨Inv.inv k, ⋯⟩) 1
    -/
    apply Subtype.ext_iff_val.2
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Padic p
      property✝ : LE.le (Norm.norm k) 1
      h : Eq (Norm.norm k) 1
      hk : Ne k 0
      ⊢ Eq ↑(HMul.hMul ⟨k, property✝⟩ ⟨Inv.inv k, ⋯⟩) ↑1
    -/
    simp [mul_inv_cancel₀ hk]
    /-
      🎉 no goals
    -/


                                                                 /-
                                                                   p : Nat
                                                                   hp : Fact (Nat.Prime p)
                                                                   z : PadicInt p
                                                                   hz : Eq (Norm.norm z) 1
                                                                   ⊢ Eq (HMul.hMul z.inv z) 1
                                                                 -/
theorem inv_mul {z : ℤ_[p]} (hz : ‖z‖ = 1) : z.inv * z = 1 := by rw [mul_comm, mul_inv hz]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem isUnit_iff {z : ℤ_[p]} : IsUnit z ↔ ‖z‖ = 1 :=
  ⟨fun h => by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      z : PadicInt p
      h : IsUnit z
      ⊢ Eq (Norm.norm z) 1
    -/
    rcases isUnit_iff_dvd_one.1 h with ⟨w, eq⟩
    /-
      case intro
      p : Nat
      hp : Fact (Nat.Prime p)
      z : PadicInt p
      h : IsUnit z
      w : PadicInt p
      eq : Eq 1 (HMul.hMul z w)
      ⊢ Eq (Norm.norm z) 1
    -/
    refine le_antisymm (norm_le_one _) ?_
    /-
      case intro
      p : Nat
      hp : Fact (Nat.Prime p)
      z : PadicInt p
      h : IsUnit z
      w : PadicInt p
      eq : Eq 1 (HMul.hMul z w)
      ⊢ LE.le 1 (Norm.norm z)
    -/
    have := mul_le_mul_of_nonneg_left (norm_le_one w) (norm_nonneg z)
    /-
      case intro
      p : Nat
      hp : Fact (Nat.Prime p)
      z : PadicInt p
      h : IsUnit z
      w : PadicInt p
      eq : Eq 1 (HMul.hMul z w)
      this : LE.le (HMul.hMul (Norm.norm z) (Norm.norm w)) (HMul.hMul (Norm.norm z) 1)
      ⊢ LE.le 1 (Norm.norm z)
    -/
    rwa [mul_one, ← norm_mul, ← eq, norm_one] at this, fun h =>
    /-
      🎉 no goals
    -/
    ⟨⟨z, z.inv, mul_inv h, inv_mul h⟩, rfl⟩⟩


theorem norm_lt_one_add {z1 z2 : ℤ_[p]} (hz1 : ‖z1‖ < 1) (hz2 : ‖z2‖ < 1) : ‖z1 + z2‖ < 1 :=
  lt_of_le_of_lt (nonarchimedean _ _) (max_lt hz1 hz2)


theorem norm_lt_one_mul {z1 z2 : ℤ_[p]} (hz2 : ‖z2‖ < 1) : ‖z1 * z2‖ < 1 :=
  calc
                                  /-
                                    p : Nat
                                    hp : Fact (Nat.Prime p)
                                    z1 z2 : PadicInt p
                                    hz2 : LT.lt (Norm.norm z2) 1
                                    ⊢ Eq (Norm.norm (HMul.hMul z1 z2)) (HMul.hMul (Norm.norm z1) (Norm.norm z2))
                                  -/
    ‖z1 * z2‖ = ‖z1‖ * ‖z2‖ := by simp
                                  /-
                                    🎉 no goals
                                  -/
    _ < 1 := mul_lt_one_of_nonneg_of_lt_one_right (norm_le_one _) (norm_nonneg _) hz2


theorem mem_nonunits {z : ℤ_[p]} : z ∈ nonunits ℤ_[p] ↔ ‖z‖ < 1 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    z : PadicInt p
    ⊢ Iff (Membership.mem (nonunits (PadicInt p)) z) (LT.lt (Norm.norm z) 1)
  -/
  rw [lt_iff_le_and_ne]; simp [norm_le_one z, nonunits, isUnit_iff]
                         /-
                           🎉 no goals
                         -/


theorem not_isUnit_iff {z : ℤ_[p]} : ¬IsUnit z ↔ ‖z‖ < 1 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    z : PadicInt p
    ⊢ Iff (Not (IsUnit z)) (LT.lt (Norm.norm z) 1)
  -/
  simpa using mem_nonunits
  /-
    🎉 no goals
  -/


/-- A `p`-adic number `u` with `‖u‖ = 1` is a unit of `ℤ_[p]`. -/
def mkUnits {u : ℚ_[p]} (h : ‖u‖ = 1) : ℤ_[p]ˣ :=
  let z : ℤ_[p] := ⟨u, le_of_eq h⟩
  ⟨z, z.inv, mul_inv h, inv_mul h⟩


@[simp]
theorem mkUnits_eq {u : ℚ_[p]} (h : ‖u‖ = 1) : ((mkUnits h : ℤ_[p]) : ℚ_[p]) = u := rfl


@[simp]
                                                                           /-
                                                                             p : Nat
                                                                             hp : Fact (Nat.Prime p)
                                                                             u : Units (PadicInt p)
                                                                             ⊢ IsUnit ↑u
                                                                           -/
theorem norm_units (u : ℤ_[p]ˣ) : ‖(u : ℤ_[p])‖ = 1 := isUnit_iff.mp <| by simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- `unitCoeff hx` is the unit `u` in the unique representation `x = u * p ^ n`.
See `unitCoeff_spec`. -/
def unitCoeff {x : ℤ_[p]} (hx : x ≠ 0) : ℤ_[p]ˣ :=
  let u : ℚ_[p] := x * (p : ℚ_[p]) ^ (-x.valuation : ℤ)
  have hu : ‖u‖ = 1 := by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      x✝ y x : PadicInt p
      hx : Ne x 0
      u : Padic p := HMul.hMul (↑x) (HPow.hPow (↑p) (Neg.neg ↑x.valuation))
      ⊢ Eq (Norm.norm u) 1
    -/
    simp [u, hx, pow_ne_zero _ (NeZero.ne _), norm_eq_zpow_neg_valuation]
    /-
      🎉 no goals
    -/
  mkUnits hu


@[simp]
theorem unitCoeff_coe {x : ℤ_[p]} (hx : x ≠ 0) :
    (unitCoeff hx : ℚ_[p]) = x * (p : ℚ_[p]) ^ (-x.valuation : ℤ) := rfl


theorem unitCoeff_spec {x : ℤ_[p]} (hx : x ≠ 0) :
    x = (unitCoeff hx : ℤ_[p]) * (p : ℤ_[p]) ^ x.valuation := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    hx : Ne x 0
    ⊢ Eq x (HMul.hMul (↑(PadicInt.unitCoeff hx)) (HPow.hPow (↑p) x.valuation))
  -/
  apply Subtype.coe_injective
  /-
    case a
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    hx : Ne x 0
    ⊢ Eq ((fun a => ↑a) x) ((fun a => ↑a) (HMul.hMul (↑(PadicInt.unitCoeff hx)) (H …
  -/
  push_cast
  /-
    case a
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    hx : Ne x 0
    ⊢ Eq (↑x) (HMul.hMul (↑↑(PadicInt.unitCoeff hx)) (HPow.hPow (↑p) x.valuation))
  -/
  rw [unitCoeff_coe, mul_assoc, ← zpow_natCast, ← zpow_add₀]
    /-
      case a
      p : Nat
      hp : Fact (Nat.Prime p)
      x : PadicInt p
      hx : Ne x 0
      ⊢ Eq (↑x) (HMul.hMul (↑x) (HPow.hPow (↑p) (HAdd.hAdd (Neg.neg ↑x.valuation) ↑x …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case a.ha
      p : Nat
      hp : Fact (Nat.Prime p)
      x : PadicInt p
      hx : Ne x 0
      ⊢ Ne (↑p) 0
    -/
  · exact NeZero.ne _
    /-
      🎉 no goals
    -/


theorem norm_le_pow_iff_le_valuation (x : ℤ_[p]) (hx : x ≠ 0) (n : ℕ) :
    ‖x‖ ≤ (p : ℝ) ^ (-n : ℤ) ↔ n ≤ x.valuation := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    hx : Ne x 0
    n : Nat
    ⊢ Iff (LE.le (Norm.norm x) (HPow.hPow (↑p) (Neg.neg ↑n))) (LE.le n x.valuation)
  -/
  rw [norm_eq_zpow_neg_valuation hx, zpow_le_zpow_iff_right₀, neg_le_neg_iff, Nat.cast_le]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    hx : Ne x 0
    n : Nat
    ⊢ LT.lt 1 ↑p
  -/
  exact mod_cast hp.out.one_lt
  /-
    🎉 no goals
  -/


theorem mem_span_pow_iff_le_valuation (x : ℤ_[p]) (hx : x ≠ 0) (n : ℕ) :
    x ∈ (Ideal.span {(p : ℤ_[p]) ^ n} : Ideal ℤ_[p]) ↔ n ≤ x.valuation := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    hx : Ne x 0
    n : Nat
    ⊢ Iff (Membership.mem (Ideal.span (Singleton.singleton (HPow.hPow (↑p) n))) x) …
  -/
  rw [Ideal.mem_span_singleton]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    hx : Ne x 0
    n : Nat
    ⊢ Iff (Dvd.dvd (HPow.hPow (↑p) n) x) (LE.le n x.valuation)
  -/
  constructor
    /-
      case mp
      p : Nat
      hp : Fact (Nat.Prime p)
      x : PadicInt p
      hx : Ne x 0
      n : Nat
      ⊢ Dvd.dvd (HPow.hPow (↑p) n) x → LE.le n x.valuation
    -/
  · rintro ⟨c, rfl⟩
    suffices c ≠ 0 by
      rw [valuation_p_pow_mul _ _ this]
      exact le_self_add
    /-
      case mp.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      c : PadicInt p
      hx : Ne (HMul.hMul (HPow.hPow (↑p) n) c) 0
      ⊢ Ne c 0
    -/
    contrapose! hx
    /-
      case mp.intro
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      c : PadicInt p
      hx : Eq c 0
      ⊢ Eq (HMul.hMul (HPow.hPow (↑p) n) c) 0
    -/
    rw [hx, mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      p : Nat
      hp : Fact (Nat.Prime p)
      x : PadicInt p
      hx : Ne x 0
      n : Nat
      ⊢ LE.le n x.valuation → Dvd.dvd (HPow.hPow (↑p) n) x
    -/
  · nth_rewrite 2 [unitCoeff_spec hx]
    /-
      case mpr
      p : Nat
      hp : Fact (Nat.Prime p)
      x : PadicInt p
      hx : Ne x 0
      n : Nat
      ⊢ LE.le n x.valuation → Dvd.dvd (HPow.hPow (↑p) n) (HMul.hMul (↑(PadicInt.unit …
    -/
    simpa [Units.isUnit, IsUnit.dvd_mul_left] using pow_dvd_pow _
    /-
      🎉 no goals
    -/


theorem norm_le_pow_iff_mem_span_pow (x : ℤ_[p]) (n : ℕ) :
    ‖x‖ ≤ (p : ℝ) ^ (-n : ℤ) ↔ x ∈ (Ideal.span {(p : ℤ_[p]) ^ n} : Ideal ℤ_[p]) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    n : Nat
    ⊢ Iff (LE.le (Norm.norm x) (HPow.hPow (↑p) (Neg.neg ↑n))) (Membership.mem (Ide …
  -/
  by_cases hx : x = 0
    /-
      case pos
      p : Nat
      hp : Fact (Nat.Prime p)
      x : PadicInt p
      n : Nat
      hx : Eq x 0
      ⊢ Iff (LE.le (Norm.norm x) (HPow.hPow (↑p) (Neg.neg ↑n))) (Membership.mem (Ide …
    -/
  · subst hx
    /-
      case pos
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ Iff (LE.le (Norm.norm 0) (HPow.hPow (↑p) (Neg.neg ↑n))) (Membership.mem (Ide …
    -/
    simp only [norm_zero, zpow_neg, zpow_natCast, inv_nonneg, iff_true, Submodule.zero_mem]
    /-
      case pos
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      ⊢ LE.le 0 (HPow.hPow (↑p) n)
    -/
    exact mod_cast Nat.zero_le _
    /-
      🎉 no goals
    -/
  /-
    case neg
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    n : Nat
    hx : Not (Eq x 0)
    ⊢ Iff (LE.le (Norm.norm x) (HPow.hPow (↑p) (Neg.neg ↑n))) (Membership.mem (Ide …
  -/
  rw [norm_le_pow_iff_le_valuation x hx, mem_span_pow_iff_le_valuation x hx]
  /-
    🎉 no goals
  -/


theorem norm_le_pow_iff_norm_lt_pow_add_one (x : ℤ_[p]) (n : ℤ) :
    ‖x‖ ≤ (p : ℝ) ^ n ↔ ‖x‖ < (p : ℝ) ^ (n + 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    n : Int
    ⊢ Iff (LE.le (Norm.norm x) (HPow.hPow (↑p) n)) (LT.lt (Norm.norm x) (HPow.hPow …
  -/
  rw [norm_def]; exact Padic.norm_le_pow_iff_norm_lt_pow_add_one _ _
                 /-
                   🎉 no goals
                 -/


theorem norm_lt_pow_iff_norm_le_pow_sub_one (x : ℤ_[p]) (n : ℤ) :
    ‖x‖ < (p : ℝ) ^ n ↔ ‖x‖ ≤ (p : ℝ) ^ (n - 1) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    n : Int
    ⊢ Iff (LT.lt (Norm.norm x) (HPow.hPow (↑p) n)) (LE.le (Norm.norm x) (HPow.hPow …
  -/
  rw [norm_le_pow_iff_norm_lt_pow_add_one, sub_add_cancel]
  /-
    🎉 no goals
  -/


theorem norm_lt_one_iff_dvd (x : ℤ_[p]) : ‖x‖ < 1 ↔ ↑p ∣ x := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    ⊢ Iff (LT.lt (Norm.norm x) 1) (Dvd.dvd (↑p) x)
  -/
  have := norm_le_pow_iff_mem_span_pow x 1
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    this : Iff (LE.le (Norm.norm x) (HPow.hPow (↑p) (Neg.neg ↑1))) (Membership.mem …
    ⊢ Iff (LT.lt (Norm.norm x) 1) (Dvd.dvd (↑p) x)
  -/
  rw [Ideal.mem_span_singleton, pow_one] at this
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    this : Iff (LE.le (Norm.norm x) (HPow.hPow (↑p) (Neg.neg ↑1))) (Dvd.dvd (↑p) x)
    ⊢ Iff (LT.lt (Norm.norm x) 1) (Dvd.dvd (↑p) x)
  -/
  rw [← this, norm_le_pow_iff_norm_lt_pow_add_one]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : PadicInt p
    this : Iff (LE.le (Norm.norm x) (HPow.hPow (↑p) (Neg.neg ↑1))) (Dvd.dvd (↑p) x)
    ⊢ Iff (LT.lt (Norm.norm x) 1) (LT.lt (Norm.norm x) (HPow.hPow (↑p) (HAdd.hAdd  …
  -/
  simp only [zpow_zero, Int.ofNat_zero, Int.ofNat_succ, neg_add_cancel, zero_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem pow_p_dvd_int_iff (n : ℕ) (a : ℤ) : (p : ℤ_[p]) ^ n ∣ a ↔ (p ^ n : ℤ) ∣ a := by
  rw [← Nat.cast_pow, ← norm_int_le_pow_iff_dvd, norm_le_pow_iff_mem_span_pow,
    Ideal.mem_span_singleton, Nat.cast_pow]


instance : IsLocalRing ℤ_[p] :=
                                    /-
                                      p : Nat
                                      hp : Fact (Nat.Prime p)
                                      x y : PadicInt p
                                      ⊢ ∀ (a b : PadicInt p), Membership.mem (nonunits (PadicInt p)) a → Membership. …
                                    -/
  IsLocalRing.of_nonunits_add <| by simp only [mem_nonunits]; exact fun x y => norm_lt_one_add
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem p_nonnunit : (p : ℤ_[p]) ∈ nonunits ℤ_[p] := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Membership.mem (nonunits (PadicInt p)) ↑p
  -/
  have : (p : ℝ)⁻¹ < 1 := inv_lt_one_of_one_lt₀ <| mod_cast hp.out.one_lt
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    this : LT.lt (Inv.inv ↑p) 1
    ⊢ Membership.mem (nonunits (PadicInt p)) ↑p
  -/
  rwa [← norm_p, ← mem_nonunits] at this
  /-
    🎉 no goals
  -/


theorem maximalIdeal_eq_span_p : maximalIdeal ℤ_[p] = Ideal.span {(p : ℤ_[p])} := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (IsLocalRing.maximalIdeal (PadicInt p)) (Ideal.span (Singleton.singleton  …
  -/
  apply le_antisymm
    /-
      case a
      p : Nat
      hp : Fact (Nat.Prime p)
      ⊢ LE.le (IsLocalRing.maximalIdeal (PadicInt p)) (Ideal.span (Singleton.singlet …
    -/
  · intro x hx
    /-
      case a
      p : Nat
      hp : Fact (Nat.Prime p)
      x : PadicInt p
      hx : Membership.mem (IsLocalRing.maximalIdeal (PadicInt p)) x
      ⊢ Membership.mem (Ideal.span (Singleton.singleton ↑p)) x
    -/
    simp only [IsLocalRing.mem_maximalIdeal, mem_nonunits] at hx
    /-
      case a
      p : Nat
      hp : Fact (Nat.Prime p)
      x : PadicInt p
      hx : LT.lt (Norm.norm x) 1
      ⊢ Membership.mem (Ideal.span (Singleton.singleton ↑p)) x
    -/
    rwa [Ideal.mem_span_singleton, ← norm_lt_one_iff_dvd]
    /-
      🎉 no goals
    -/
    /-
      case a
      p : Nat
      hp : Fact (Nat.Prime p)
      ⊢ LE.le (Ideal.span (Singleton.singleton ↑p)) (IsLocalRing.maximalIdeal (Padic …
    -/
  · rw [Ideal.span_le, Set.singleton_subset_iff]
    /-
      case a
      p : Nat
      hp : Fact (Nat.Prime p)
      ⊢ Membership.mem ↑(IsLocalRing.maximalIdeal (PadicInt p)) ↑p
    -/
    exact p_nonnunit
    /-
      🎉 no goals
    -/


theorem prime_p : Prime (p : ℤ_[p]) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Prime ↑p
  -/
  rw [← Ideal.span_singleton_prime, ← maximalIdeal_eq_span_p]
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      ⊢ (IsLocalRing.maximalIdeal (PadicInt p)).IsPrime
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      ⊢ Ne (↑p) 0
    -/
  · exact NeZero.ne _
    /-
      🎉 no goals
    -/


theorem irreducible_p : Irreducible (p : ℤ_[p]) := Prime.irreducible prime_p


instance : IsDiscreteValuationRing ℤ_[p] :=
  IsDiscreteValuationRing.ofHasUnitMulPowIrreducibleFactorization
    ⟨p, irreducible_p, fun {x hx} =>
                                     /-
                                       p : Nat
                                       hp : Fact (Nat.Prime p)
                                       x✝ y x : PadicInt p
                                       hx : Ne x 0
                                       ⊢ Eq (HMul.hMul (HPow.hPow (↑p) x.valuation) ↑(PadicInt.unitCoeff hx)) x
                                     -/
      ⟨x.valuation, unitCoeff hx, by rw [mul_comm, ← unitCoeff_spec hx]⟩⟩
                                     /-
                                       🎉 no goals
                                     -/


theorem ideal_eq_span_pow_p {s : Ideal ℤ_[p]} (hs : s ≠ ⊥) :
    ∃ n : ℕ, s = Ideal.span {(p : ℤ_[p]) ^ n} :=
  IsDiscreteValuationRing.ideal_eq_span_pow_irreducible hs irreducible_p


instance : IsAdicComplete (maximalIdeal ℤ_[p]) ℤ_[p] where
  prec' x hx := by
    simp only [← Ideal.one_eq_top, smul_eq_mul, mul_one, SModEq.sub_mem, maximalIdeal_eq_span_p,
      Ideal.span_singleton_pow, ← norm_le_pow_iff_mem_span_pow] at hx ⊢
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      x✝ y : PadicInt p
      x : Nat → PadicInt p
      hx : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HPo …
      ⊢ Exists fun L => ∀ (n : Nat), LE.le (Norm.norm (HSub.hSub (x n) L)) (HPow.hPo …
    -/
    let x' : CauSeq ℤ_[p] norm := ⟨x, ?_⟩; swap
      /-
        case refine_1
        p : Nat
        hp : Fact (Nat.Prime p)
        x✝ y : PadicInt p
        x : Nat → PadicInt p
        hx : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HPo …
        ⊢ IsCauSeq Norm.norm x
      -/
    · intro ε hε
      /-
        case refine_1
        p : Nat
        hp : Fact (Nat.Prime p)
        x✝ y : PadicInt p
        x : Nat → PadicInt p
        hx : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HPo …
        ε : Real
        hε : GT.gt ε 0
        ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (Norm.norm (HSub.hSub (x j) ( …
      -/
      obtain ⟨m, hm⟩ := exists_pow_neg_lt p hε
      /-
        case refine_1.intro
        p : Nat
        hp : Fact (Nat.Prime p)
        x✝ y : PadicInt p
        x : Nat → PadicInt p
        hx : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HPo …
        ε : Real
        hε : GT.gt ε 0
        m : Nat
        hm : LT.lt (HPow.hPow (↑p) (Neg.neg ↑m)) ε
        ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (Norm.norm (HSub.hSub (x j) ( …
      -/
      refine ⟨m, fun n hn => lt_of_le_of_lt ?_ hm⟩
      /-
        case refine_1.intro
        p : Nat
        hp : Fact (Nat.Prime p)
        x✝ y : PadicInt p
        x : Nat → PadicInt p
        hx : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HPo …
        ε : Real
        hε : GT.gt ε 0
        m : Nat
        hm : LT.lt (HPow.hPow (↑p) (Neg.neg ↑m)) ε
        n : Nat
        hn : GE.ge n m
        ⊢ LE.le (Norm.norm (HSub.hSub (x n) (x m))) (HPow.hPow (↑p) (Neg.neg ↑m))
      -/
      rw [← neg_sub, norm_neg]
      /-
        case refine_1.intro
        p : Nat
        hp : Fact (Nat.Prime p)
        x✝ y : PadicInt p
        x : Nat → PadicInt p
        hx : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HPo …
        ε : Real
        hε : GT.gt ε 0
        m : Nat
        hm : LT.lt (HPow.hPow (↑p) (Neg.neg ↑m)) ε
        n : Nat
        hn : GE.ge n m
        ⊢ LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HPow.hPow (↑p) (Neg.neg ↑m))
      -/
      exact hx hn
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        p : Nat
        hp : Fact (Nat.Prime p)
        x✝ y : PadicInt p
        x : Nat → PadicInt p
        hx : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HPo …
        x' : CauSeq (PadicInt p) Norm.norm := ⟨x, ⋯⟩
        ⊢ Exists fun L => ∀ (n : Nat), LE.le (Norm.norm (HSub.hSub (x n) L)) (HPow.hPo …
      -/
    · refine ⟨x'.lim, fun n => ?_⟩
      /-
        case refine_2
        p : Nat
        hp : Fact (Nat.Prime p)
        x✝ y : PadicInt p
        x : Nat → PadicInt p
        hx : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HPo …
        x' : CauSeq (PadicInt p) Norm.norm := ⟨x, ⋯⟩
        n : Nat
        ⊢ LE.le (Norm.norm (HSub.hSub (x n) x'.lim)) (HPow.hPow (↑p) (Neg.neg ↑n))
      -/
      have : (0 : ℝ) < (p : ℝ) ^ (-n : ℤ) := zpow_pos (mod_cast hp.out.pos) _
      /-
        case refine_2
        p : Nat
        hp : Fact (Nat.Prime p)
        x✝ y : PadicInt p
        x : Nat → PadicInt p
        hx : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HPo …
        x' : CauSeq (PadicInt p) Norm.norm := ⟨x, ⋯⟩
        n : Nat
        this : LT.lt 0 (HPow.hPow (↑p) (Neg.neg ↑n))
        ⊢ LE.le (Norm.norm (HSub.hSub (x n) x'.lim)) (HPow.hPow (↑p) (Neg.neg ↑n))
      -/
      obtain ⟨i, hi⟩ := equiv_def₃ (equiv_lim x') this
      /-
        case refine_2.intro
        p : Nat
        hp : Fact (Nat.Prime p)
        x✝ y : PadicInt p
        x : Nat → PadicInt p
        hx : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HPo …
        x' : CauSeq (PadicInt p) Norm.norm := ⟨x, ⋯⟩
        n : Nat
        this : LT.lt 0 (HPow.hPow (↑p) (Neg.neg ↑n))
        i : Nat
        hi : ∀ (j : Nat), GE.ge j i → ∀ (k : Nat), GE.ge k j → LT.lt (Norm.norm (HSub. …
        ⊢ LE.le (Norm.norm (HSub.hSub (x n) x'.lim)) (HPow.hPow (↑p) (Neg.neg ↑n))
      -/
      by_cases hin : i ≤ n
        /-
          case pos
          p : Nat
          hp : Fact (Nat.Prime p)
          x✝ y : PadicInt p
          x : Nat → PadicInt p
          hx : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HPo …
          x' : CauSeq (PadicInt p) Norm.norm := ⟨x, ⋯⟩
          n : Nat
          this : LT.lt 0 (HPow.hPow (↑p) (Neg.neg ↑n))
          i : Nat
          hi : ∀ (j : Nat), GE.ge j i → ∀ (k : Nat), GE.ge k j → LT.lt (Norm.norm (HSub. …
          hin : LE.le i n
          ⊢ LE.le (Norm.norm (HSub.hSub (x n) x'.lim)) (HPow.hPow (↑p) (Neg.neg ↑n))
        -/
      · exact (hi i le_rfl n hin).le
        /-
          🎉 no goals
        -/
        /-
          case neg
          p : Nat
          hp : Fact (Nat.Prime p)
          x✝ y : PadicInt p
          x : Nat → PadicInt p
          hx : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HPo …
          x' : CauSeq (PadicInt p) Norm.norm := ⟨x, ⋯⟩
          n : Nat
          this : LT.lt 0 (HPow.hPow (↑p) (Neg.neg ↑n))
          i : Nat
          hi : ∀ (j : Nat), GE.ge j i → ∀ (k : Nat), GE.ge k j → LT.lt (Norm.norm (HSub. …
          hin : Not (LE.le i n)
          ⊢ LE.le (Norm.norm (HSub.hSub (x n) x'.lim)) (HPow.hPow (↑p) (Neg.neg ↑n))
        -/
      · push_neg at hin
        /-
          case neg
          p : Nat
          hp : Fact (Nat.Prime p)
          x✝ y : PadicInt p
          x : Nat → PadicInt p
          hx : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HPo …
          x' : CauSeq (PadicInt p) Norm.norm := ⟨x, ⋯⟩
          n : Nat
          this : LT.lt 0 (HPow.hPow (↑p) (Neg.neg ↑n))
          i : Nat
          hi : ∀ (j : Nat), GE.ge j i → ∀ (k : Nat), GE.ge k j → LT.lt (Norm.norm (HSub. …
          hin : LT.lt n i
          ⊢ LE.le (Norm.norm (HSub.hSub (x n) x'.lim)) (HPow.hPow (↑p) (Neg.neg ↑n))
        -/
        specialize hi i le_rfl i le_rfl
        /-
          case neg
          p : Nat
          hp : Fact (Nat.Prime p)
          x✝ y : PadicInt p
          x : Nat → PadicInt p
          hx : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HPo …
          x' : CauSeq (PadicInt p) Norm.norm := ⟨x, ⋯⟩
          n : Nat
          this : LT.lt 0 (HPow.hPow (↑p) (Neg.neg ↑n))
          i : Nat
          hin : LT.lt n i
          hi : LT.lt (Norm.norm (HSub.hSub (↑x' i) (↑(CauSeq.const Norm.norm x'.lim) i)) …
          ⊢ LE.le (Norm.norm (HSub.hSub (x n) x'.lim)) (HPow.hPow (↑p) (Neg.neg ↑n))
        -/
        specialize hx hin.le
        /-
          case neg
          p : Nat
          hp : Fact (Nat.Prime p)
          x✝ y : PadicInt p
          x : Nat → PadicInt p
          hx✝ : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HP …
          x' : CauSeq (PadicInt p) Norm.norm := ⟨x, ⋯⟩
          n : Nat
          this : LT.lt 0 (HPow.hPow (↑p) (Neg.neg ↑n))
          i : Nat
          hin : LT.lt n i
          hi : LT.lt (Norm.norm (HSub.hSub (↑x' i) (↑(CauSeq.const Norm.norm x'.lim) i)) …
          hx : LE.le (Norm.norm (HSub.hSub (x n) (x i))) (HPow.hPow (↑p) (Neg.neg ↑n))
          ⊢ LE.le (Norm.norm (HSub.hSub (x n) x'.lim)) (HPow.hPow (↑p) (Neg.neg ↑n))
        -/
        have := nonarchimedean (x n - x i : ℤ_[p]) (x i - x'.lim)
        /-
          case neg
          p : Nat
          hp : Fact (Nat.Prime p)
          x✝ y : PadicInt p
          x : Nat → PadicInt p
          hx✝ : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HP …
          x' : CauSeq (PadicInt p) Norm.norm := ⟨x, ⋯⟩
          n : Nat
          this✝ : LT.lt 0 (HPow.hPow (↑p) (Neg.neg ↑n))
          i : Nat
          hin : LT.lt n i
          hi : LT.lt (Norm.norm (HSub.hSub (↑x' i) (↑(CauSeq.const Norm.norm x'.lim) i)) …
          hx : LE.le (Norm.norm (HSub.hSub (x n) (x i))) (HPow.hPow (↑p) (Neg.neg ↑n))
          this : LE.le (Norm.norm (HAdd.hAdd (HSub.hSub (x n) (x i)) (HSub.hSub (x i) x' …
          ⊢ LE.le (Norm.norm (HSub.hSub (x n) x'.lim)) (HPow.hPow (↑p) (Neg.neg ↑n))
        -/
        rw [sub_add_sub_cancel] at this
        /-
          case neg
          p : Nat
          hp : Fact (Nat.Prime p)
          x✝ y : PadicInt p
          x : Nat → PadicInt p
          hx✝ : ∀ {m n : Nat}, LE.le m n → LE.le (Norm.norm (HSub.hSub (x m) (x n))) (HP …
          x' : CauSeq (PadicInt p) Norm.norm := ⟨x, ⋯⟩
          n : Nat
          this✝ : LT.lt 0 (HPow.hPow (↑p) (Neg.neg ↑n))
          i : Nat
          hin : LT.lt n i
          hi : LT.lt (Norm.norm (HSub.hSub (↑x' i) (↑(CauSeq.const Norm.norm x'.lim) i)) …
          hx : LE.le (Norm.norm (HSub.hSub (x n) (x i))) (HPow.hPow (↑p) (Neg.neg ↑n))
          this : LE.le (Norm.norm (HSub.hSub (x n) x'.lim)) (Max.max (Norm.norm (HSub.hS …
          ⊢ LE.le (Norm.norm (HSub.hSub (x n) x'.lim)) (HPow.hPow (↑p) (Neg.neg ↑n))
        -/
        exact this.trans (max_le_iff.mpr ⟨hx, hi.le⟩)
        /-
          🎉 no goals
        -/


instance algebra : Algebra ℤ_[p] ℚ_[p] :=
  Algebra.ofSubring (subring p)


@[simp]
theorem algebraMap_apply (x : ℤ_[p]) : algebraMap ℤ_[p] ℚ_[p] x = x :=
  rfl


instance isFractionRing : IsFractionRing ℤ_[p] ℚ_[p] where
  map_units' := fun ⟨x, hx⟩ => by
    rwa [algebraMap_apply, isUnit_iff_ne_zero, PadicInt.coe_ne_zero, ←
      mem_nonZeroDivisors_iff_ne_zero]
  surj' x := by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      x✝ y : PadicInt p
      x : Padic p
      ⊢ Exists fun x_1 => Eq (HMul.hMul x ((algebraMap (PadicInt p) (Padic p)) ↑x_1. …
    -/
    by_cases hx : ‖x‖ ≤ 1
      /-
        case pos
        p : Nat
        hp : Fact (Nat.Prime p)
        x✝ y : PadicInt p
        x : Padic p
        hx : LE.le (Norm.norm x) 1
        ⊢ Exists fun x_1 => Eq (HMul.hMul x ((algebraMap (PadicInt p) (Padic p)) ↑x_1. …
      -/
    · use (⟨x, hx⟩, 1)
      /-
        case h
        p : Nat
        hp : Fact (Nat.Prime p)
        x✝ y : PadicInt p
        x : Padic p
        hx : LE.le (Norm.norm x) 1
        ⊢ Eq (HMul.hMul x ((algebraMap (PadicInt p) (Padic p)) ↑{ fst := ⟨x, hx⟩, snd  …
      -/
      rw [Submonoid.coe_one, map_one, mul_one, PadicInt.algebraMap_apply, Subtype.coe_mk]
      /-
        🎉 no goals
      -/
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        x✝ y : PadicInt p
        x : Padic p
        hx : Not (LE.le (Norm.norm x) 1)
        ⊢ Exists fun x_1 => Eq (HMul.hMul x ((algebraMap (PadicInt p) (Padic p)) ↑x_1. …
      -/
    · set n := Int.toNat (-x.valuation) with hn
      have hn_coe : (n : ℤ) = -x.valuation := by
        rw [hn, Int.toNat_of_nonneg]
        rw [Right.nonneg_neg_iff]
        rw [Padic.norm_le_one_iff_val_nonneg, not_le] at hx
        exact hx.le
      /-
        case neg
        p : Nat
        hp : Fact (Nat.Prime p)
        x✝ y : PadicInt p
        x : Padic p
        hx : Not (LE.le (Norm.norm x) 1)
        n : Nat := (Neg.neg x.valuation).toNat
        hn : Eq n (Neg.neg x.valuation).toNat
        hn_coe : Eq (↑n) (Neg.neg x.valuation)
        ⊢ Exists fun x_1 => Eq (HMul.hMul x ((algebraMap (PadicInt p) (Padic p)) ↑x_1. …
      -/
      set a := x * (p : ℚ_[p]) ^ n with ha
      have ha_norm : ‖a‖ = 1 := by
        have hx : x ≠ 0 := by
          intro h0
          rw [h0, norm_zero] at hx
          exact hx zero_le_one
        rw [ha, padicNormE.mul, padicNormE.norm_p_pow, Padic.norm_eq_zpow_neg_valuation hx,
          ← zpow_add', hn_coe, neg_neg, neg_add_cancel, zpow_zero]
        exact Or.inl (Nat.cast_ne_zero.mpr (NeZero.ne p))
      use
        (⟨a, le_of_eq ha_norm⟩,
          ⟨(p ^ n : ℤ_[p]), mem_nonZeroDivisors_iff_ne_zero.mpr (NeZero.ne _)⟩)
      simp only [a, map_pow, map_natCast, algebraMap_apply, PadicInt.coe_pow,
        PadicInt.coe_natCast, Subtype.coe_mk, Nat.cast_pow]
  exists_of_eq := by
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      x y : PadicInt p
      ⊢ ∀ {x y : PadicInt p}, Eq ((algebraMap (PadicInt p) (Padic p)) x) ((algebraMa …
    -/
    simp_rw [algebraMap_apply, Subtype.coe_inj]
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      x y : PadicInt p
      ⊢ ∀ {x y : PadicInt p}, Eq x y → Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.h …
    -/
    exact fun h => ⟨1, by rw [h]⟩
    /-
      🎉 no goals
    -/


