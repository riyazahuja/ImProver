/-- Grade involution, inverting the sign of each basis vector. -/
def involute : CliffordAlgebra Q →ₐ[R] CliffordAlgebra Q :=
                                            /-
                                              R : Type u_1
                                              inst✝² : CommRing R
                                              M : Type u_2
                                              inst✝¹ : AddCommGroup M
                                              inst✝ : Module R M
                                              Q : QuadraticForm R M
                                              m : M
                                              ⊢ Eq (HMul.hMul ((Neg.neg (CliffordAlgebra.ι Q)) m) ((Neg.neg (CliffordAlgebra …
                                            -/
  CliffordAlgebra.lift Q ⟨-ι Q, fun m => by simp⟩
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem involute_ι (m : M) : involute (ι Q m) = -ι Q m :=
  lift_ι_apply _ _ m


@[simp]
theorem involute_comp_involute : involute.comp involute = AlgHom.id R (CliffordAlgebra Q) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    ⊢ Eq (CliffordAlgebra.involute.comp CliffordAlgebra.involute) (AlgHom.id R (Cl …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem involute_involutive : Function.Involutive (involute : _ → CliffordAlgebra Q) :=
  AlgHom.congr_fun involute_comp_involute


@[simp]
theorem involute_involute : ∀ a : CliffordAlgebra Q, involute (involute a) = a :=
  involute_involutive


/-- `CliffordAlgebra.involute` as an `AlgEquiv`. -/
@[simps!]
def involuteEquiv : CliffordAlgebra Q ≃ₐ[R] CliffordAlgebra Q :=
  AlgEquiv.ofAlgHom involute involute (AlgHom.ext <| involute_involute)
    (AlgHom.ext <| involute_involute)


/-- `CliffordAlgebra.reverse` as an `AlgHom` to the opposite algebra -/
def reverseOp : CliffordAlgebra Q →ₐ[R] (CliffordAlgebra Q)ᵐᵒᵖ :=
  CliffordAlgebra.lift Q
                                                                                     /-
                                                                                       R : Type u_1
                                                                                       inst✝² : CommRing R
                                                                                       M : Type u_2
                                                                                       inst✝¹ : AddCommGroup M
                                                                                       inst✝ : Module R M
                                                                                       Q : QuadraticForm R M
                                                                                       m : M
                                                                                       ⊢ Eq (MulOpposite.unop (HMul.hMul (((↑(MulOpposite.opLinearEquiv R)).comp (Cli …
                                                                                     -/
    ⟨(MulOpposite.opLinearEquiv R).toLinearMap ∘ₗ ι Q, fun m => unop_injective <| by simp⟩
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp]
theorem reverseOp_ι (m : M) : reverseOp (ι Q m) = op (ι Q m) := lift_ι_apply _ _ _


/-- `CliffordAlgebra.reverseEquiv` as an `AlgEquiv` to the opposite algebra -/
@[simps! apply]
def reverseOpEquiv : CliffordAlgebra Q ≃ₐ[R] (CliffordAlgebra Q)ᵐᵒᵖ :=
  AlgEquiv.ofAlgHom reverseOp (AlgHom.opComm reverseOp)
                                                                   /-
                                                                     R : Type u_1
                                                                     inst✝² : CommRing R
                                                                     M : Type u_2
                                                                     inst✝¹ : AddCommGroup M
                                                                     inst✝ : Module R M
                                                                     Q : QuadraticForm R M
                                                                     x✝ : M
                                                                     ⊢ Eq (((AlgHom.unop (CliffordAlgebra.reverseOp.comp (AlgHom.opComm CliffordAlg …
                                                                   -/
    (AlgHom.unop.injective <| hom_ext <| LinearMap.ext fun _ => by simp)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                          /-
                                            R : Type u_1
                                            inst✝² : CommRing R
                                            M : Type u_2
                                            inst✝¹ : AddCommGroup M
                                            inst✝ : Module R M
                                            Q : QuadraticForm R M
                                            x✝ : M
                                            ⊢ Eq ((((AlgHom.opComm CliffordAlgebra.reverseOp).comp CliffordAlgebra.reverse …
                                          -/
    (hom_ext <| LinearMap.ext fun _ => by simp)
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem reverseOpEquiv_opComm :
    AlgEquiv.opComm (reverseOpEquiv (Q := Q)) = reverseOpEquiv.symm := rfl


/-- Grade reversion, inverting the multiplication order of basis vectors.
Also called *transpose* in some literature. -/
def reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q :=
  (opLinearEquiv R).symm.toLinearMap.comp reverseOp.toLinearMap


@[simp] theorem unop_reverseOp (x : CliffordAlgebra Q) : (reverseOp x).unop = reverse x := rfl


@[simp] theorem op_reverse (x : CliffordAlgebra Q) : op (reverse x) = reverseOp x := rfl


@[simp]
                                                          /-
                                                            R : Type u_1
                                                            inst✝² : CommRing R
                                                            M : Type u_2
                                                            inst✝¹ : AddCommGroup M
                                                            inst✝ : Module R M
                                                            Q : QuadraticForm R M
                                                            m : M
                                                            ⊢ Eq (CliffordAlgebra.reverse ((CliffordAlgebra.ι Q) m)) ((CliffordAlgebra.ι Q …
                                                          -/
theorem reverse_ι (m : M) : reverse (ι Q m) = ι Q m := by simp [reverse]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem reverse.commutes (r : R) :
    reverse (algebraMap R (CliffordAlgebra Q) r) = algebraMap R _ r :=
  op_injective <| reverseOp.commutes r


@[simp]
theorem reverse.map_one : reverse (1 : CliffordAlgebra Q) = 1 :=
  op_injective (_root_.map_one reverseOp)


@[simp]
theorem reverse.map_mul (a b : CliffordAlgebra Q) :
    reverse (a * b) = reverse b * reverse a :=
  op_injective (_root_.map_mul reverseOp a b)


@[simp]
theorem reverse_involutive : Function.Involutive (reverse (Q := Q)) :=
  AlgHom.congr_fun reverseOpEquiv.symm_comp


@[simp]
theorem reverse_comp_reverse :
    reverse.comp reverse = (LinearMap.id : _ →ₗ[R] CliffordAlgebra Q) :=
  LinearMap.ext reverse_involutive


@[simp]
theorem reverse_reverse : ∀ a : CliffordAlgebra Q, reverse (reverse a) = a :=
  reverse_involutive


/-- `CliffordAlgebra.reverse` as a `LinearEquiv`. -/
@[simps!]
def reverseEquiv : CliffordAlgebra Q ≃ₗ[R] CliffordAlgebra Q :=
  LinearEquiv.ofInvolutive reverse reverse_involutive


theorem reverse_comp_involute :
    reverse.comp involute.toLinearMap =
      (involute.toLinearMap.comp reverse : _ →ₗ[R] CliffordAlgebra Q) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    ⊢ Eq (CliffordAlgebra.reverse.comp CliffordAlgebra.involute.toLinearMap) (Clif …
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    ⊢ Eq ((CliffordAlgebra.reverse.comp CliffordAlgebra.involute.toLinearMap) x) ( …
  -/
  simp only [LinearMap.comp_apply, AlgHom.toLinearMap_apply]
  induction x using CliffordAlgebra.induction with
  | algebraMap => simp
  | ι => simp
  | mul a b ha hb => simp only [ha, hb, reverse.map_mul, map_mul]
  | add a b ha hb => simp only [ha, hb, reverse.map_add, map_add]


/-- `CliffordAlgebra.reverse` and `CliffordAlgebra.involute` commute. Note that the composition
is sometimes referred to as the "clifford conjugate". -/
theorem reverse_involute_commute : Function.Commute (reverse (Q := Q)) involute :=
  LinearMap.congr_fun reverse_comp_involute


theorem reverse_involute :
    ∀ a : CliffordAlgebra Q, reverse (involute a) = involute (reverse a) :=
  reverse_involute_commute


/-- Taking the reverse of the product a list of $n$ vectors lifted via `ι` is equivalent to
taking the product of the reverse of that list. -/
theorem reverse_prod_map_ι :
    ∀ l : List M, reverse (l.map <| ι Q).prod = (l.map <| ι Q).reverse.prod
             /-
               R : Type u_1
               inst✝² : CommRing R
               M : Type u_2
               inst✝¹ : AddCommGroup M
               inst✝ : Module R M
               Q : QuadraticForm R M
               ⊢ Eq (CliffordAlgebra.reverse (List.map (⇑(CliffordAlgebra.ι Q)) List.nil).pro …
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
                /-
                  R : Type u_1
                  inst✝² : CommRing R
                  M : Type u_2
                  inst✝¹ : AddCommGroup M
                  inst✝ : Module R M
                  Q : QuadraticForm R M
                  x : M
                  xs : List M
                  ⊢ Eq (CliffordAlgebra.reverse (List.map (⇑(CliffordAlgebra.ι Q)) (List.cons x  …
                -/
  | x::xs => by simp [reverse_prod_map_ι xs]
                /-
                  🎉 no goals
                -/


/-- Taking the involute of the product a list of $n$ vectors lifted via `ι` is equivalent to
premultiplying by ${-1}^n$. -/
theorem involute_prod_map_ι :
    ∀ l : List M, involute (l.map <| ι Q).prod = (-1 : R) ^ l.length • (l.map <| ι Q).prod
             /-
               R : Type u_1
               inst✝² : CommRing R
               M : Type u_2
               inst✝¹ : AddCommGroup M
               inst✝ : Module R M
               Q : QuadraticForm R M
               ⊢ Eq (CliffordAlgebra.involute (List.map (⇑(CliffordAlgebra.ι Q)) List.nil).pr …
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
                /-
                  R : Type u_1
                  inst✝² : CommRing R
                  M : Type u_2
                  inst✝¹ : AddCommGroup M
                  inst✝ : Module R M
                  Q : QuadraticForm R M
                  x : M
                  xs : List M
                  ⊢ Eq (CliffordAlgebra.involute (List.map (⇑(CliffordAlgebra.ι Q)) (List.cons x …
                -/
  | x::xs => by simp [pow_succ, involute_prod_map_ι xs]
                /-
                  🎉 no goals
                -/


theorem submodule_map_involute_eq_comap (p : Submodule R (CliffordAlgebra Q)) :
    p.map (involute : CliffordAlgebra Q →ₐ[R] CliffordAlgebra Q).toLinearMap =
      p.comap (involute : CliffordAlgebra Q →ₐ[R] CliffordAlgebra Q).toLinearMap :=
  Submodule.map_equiv_eq_comap_symm involuteEquiv.toLinearEquiv _


@[simp]
theorem ι_range_map_involute :
    (ι Q).range.map (involute : CliffordAlgebra Q →ₐ[R] CliffordAlgebra Q).toLinearMap =
      LinearMap.range (ι Q) :=
  (ι_range_map_lift _ _).trans (LinearMap.range_neg _)


@[simp]
theorem ι_range_comap_involute :
    (ι Q).range.comap (involute : CliffordAlgebra Q →ₐ[R] CliffordAlgebra Q).toLinearMap =
      LinearMap.range (ι Q) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    ⊢ Eq (Submodule.comap CliffordAlgebra.involute.toLinearMap (LinearMap.range (C …
  -/
  rw [← submodule_map_involute_eq_comap, ι_range_map_involute]
  /-
    🎉 no goals
  -/


@[simp]
theorem evenOdd_map_involute (n : ZMod 2) :
    (evenOdd Q n).map (involute : CliffordAlgebra Q →ₐ[R] CliffordAlgebra Q).toLinearMap =
      evenOdd Q n := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    n : ZMod 2
    ⊢ Eq (Submodule.map CliffordAlgebra.involute.toLinearMap (CliffordAlgebra.even …
  -/
  simp_rw [evenOdd, Submodule.map_iSup, Submodule.map_pow, ι_range_map_involute]
  /-
    🎉 no goals
  -/


@[simp]
theorem evenOdd_comap_involute (n : ZMod 2) :
    (evenOdd Q n).comap (involute : CliffordAlgebra Q →ₐ[R] CliffordAlgebra Q).toLinearMap =
      evenOdd Q n := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    n : ZMod 2
    ⊢ Eq (Submodule.comap CliffordAlgebra.involute.toLinearMap (CliffordAlgebra.ev …
  -/
  rw [← submodule_map_involute_eq_comap, evenOdd_map_involute]
  /-
    🎉 no goals
  -/


theorem submodule_map_reverse_eq_comap (p : Submodule R (CliffordAlgebra Q)) :
    p.map (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q) =
      p.comap (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q) :=
  Submodule.map_equiv_eq_comap_symm (reverseEquiv : _ ≃ₗ[R] _) _


@[simp]
theorem ι_range_map_reverse :
    (ι Q).range.map (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q)
      = LinearMap.range (ι Q) := by
  rw [reverse, reverseOp, Submodule.map_comp, ι_range_map_lift, LinearMap.range_comp,
    ← Submodule.map_comp]
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    ⊢ Eq (Submodule.map ((↑(MulOpposite.opLinearEquiv R).symm).comp ↑(MulOpposite. …
  -/
  exact Submodule.map_id _
  /-
    🎉 no goals
  -/


@[simp]
theorem ι_range_comap_reverse :
    (ι Q).range.comap (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q)
      = LinearMap.range (ι Q) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    ⊢ Eq (Submodule.comap CliffordAlgebra.reverse (LinearMap.range (CliffordAlgebr …
  -/
  rw [← submodule_map_reverse_eq_comap, ι_range_map_reverse]
  /-
    🎉 no goals
  -/


/-- Like `Submodule.map_mul`, but with the multiplication reversed. -/
theorem submodule_map_mul_reverse (p q : Submodule R (CliffordAlgebra Q)) :
    (p * q).map (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q) =
      q.map (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q) *
        p.map (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    p q : Submodule R (CliffordAlgebra Q)
    ⊢ Eq (Submodule.map CliffordAlgebra.reverse (HMul.hMul p q)) (HMul.hMul (Submo …
  -/
  simp_rw [reverse, Submodule.map_comp, Submodule.map_mul, Submodule.map_unop_mul]
  /-
    🎉 no goals
  -/


theorem submodule_comap_mul_reverse (p q : Submodule R (CliffordAlgebra Q)) :
    (p * q).comap (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q) =
      q.comap (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q) *
        p.comap (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    p q : Submodule R (CliffordAlgebra Q)
    ⊢ Eq (Submodule.comap CliffordAlgebra.reverse (HMul.hMul p q)) (HMul.hMul (Sub …
  -/
  simp_rw [← submodule_map_reverse_eq_comap, submodule_map_mul_reverse]
  /-
    🎉 no goals
  -/


/-- Like `Submodule.map_pow` -/
theorem submodule_map_pow_reverse (p : Submodule R (CliffordAlgebra Q)) (n : ℕ) :
    (p ^ n).map (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q) =
      p.map (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q) ^ n := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    p : Submodule R (CliffordAlgebra Q)
    n : Nat
    ⊢ Eq (Submodule.map CliffordAlgebra.reverse (HPow.hPow p n)) (HPow.hPow (Submo …
  -/
  simp_rw [reverse, Submodule.map_comp, Submodule.map_pow, Submodule.map_unop_pow]
  /-
    🎉 no goals
  -/


theorem submodule_comap_pow_reverse (p : Submodule R (CliffordAlgebra Q)) (n : ℕ) :
    (p ^ n).comap (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q) =
      p.comap (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q) ^ n := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    p : Submodule R (CliffordAlgebra Q)
    n : Nat
    ⊢ Eq (Submodule.comap CliffordAlgebra.reverse (HPow.hPow p n)) (HPow.hPow (Sub …
  -/
  simp_rw [← submodule_map_reverse_eq_comap, submodule_map_pow_reverse]
  /-
    🎉 no goals
  -/


@[simp]
theorem evenOdd_map_reverse (n : ZMod 2) :
    (evenOdd Q n).map (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q) = evenOdd Q n := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    n : ZMod 2
    ⊢ Eq (Submodule.map CliffordAlgebra.reverse (CliffordAlgebra.evenOdd Q n)) (Cl …
  -/
  simp_rw [evenOdd, Submodule.map_iSup, submodule_map_pow_reverse, ι_range_map_reverse]
  /-
    🎉 no goals
  -/


@[simp]
theorem evenOdd_comap_reverse (n : ZMod 2) :
    (evenOdd Q n).comap (reverse : CliffordAlgebra Q →ₗ[R] CliffordAlgebra Q) = evenOdd Q n := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    n : ZMod 2
    ⊢ Eq (Submodule.comap CliffordAlgebra.reverse (CliffordAlgebra.evenOdd Q n)) ( …
  -/
  rw [← submodule_map_reverse_eq_comap, evenOdd_map_reverse]
  /-
    🎉 no goals
  -/


@[simp]
theorem involute_mem_evenOdd_iff {x : CliffordAlgebra Q} {n : ZMod 2} :
    involute x ∈ evenOdd Q n ↔ x ∈ evenOdd Q n :=
  SetLike.ext_iff.mp (evenOdd_comap_involute Q n) x


@[simp]
theorem reverse_mem_evenOdd_iff {x : CliffordAlgebra Q} {n : ZMod 2} :
    reverse x ∈ evenOdd Q n ↔ x ∈ evenOdd Q n :=
  SetLike.ext_iff.mp (evenOdd_comap_reverse Q n) x


theorem involute_eq_of_mem_even {x : CliffordAlgebra Q} (h : x ∈ evenOdd Q 0) : involute x = x := by
  induction x, h using even_induction with
  | algebraMap r => exact AlgHom.commutes _ _
  | add x y _hx _hy ihx ihy =>
    rw [map_add, ihx, ihy]
  | ι_mul_ι_mul m₁ m₂ x _hx ihx =>
    rw [map_mul, map_mul, involute_ι, involute_ι, ihx, neg_mul_neg]


theorem involute_eq_of_mem_odd {x : CliffordAlgebra Q} (h : x ∈ evenOdd Q 1) : involute x = -x := by
  induction x, h using odd_induction with
  | ι m => exact involute_ι _
  | add x y _hx _hy ihx ihy =>
    rw [map_add, ihx, ihy, neg_add]
  | ι_mul_ι_mul m₁ m₂ x _hx ihx =>
    rw [map_mul, map_mul, involute_ι, involute_ι, ihx, neg_mul_neg, mul_neg]


