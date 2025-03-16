/-- `dotProduct v w` is the sum of the entrywise products `v i * w i` -/
def dotProduct [Mul α] [AddCommMonoid α] (v w : m → α) : α :=
  ∑ i, v i * w i


@[deprecated (since := "2024-12-12")] protected alias Matrix.dotProduct := dotProduct

/- The precedence of 72 comes immediately after ` • ` for `SMul.smul`,
   so that `r₁ • a ⬝ᵥ r₂ • b` is parsed as `(r₁ • a) ⬝ᵥ (r₂ • b)` here. -/

@[inherit_doc]
infixl:72 " ⬝ᵥ " => dotProduct


theorem dotProduct_assoc [NonUnitalSemiring α] (u : m → α) (w : n → α) (v : Matrix m n α) :
    (fun j => u ⬝ᵥ fun i => v i j) ⬝ᵥ w = u ⬝ᵥ fun i => v i ⬝ᵥ w := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : NonUnitalSemiring α
    u : m → α
    w : n → α
    v : Matrix m n α
    ⊢ Eq (dotProduct (fun j => dotProduct u fun i => v i j) w) (dotProduct u fun i …
  -/
  simpa [dotProduct, Finset.mul_sum, Finset.sum_mul, mul_assoc] using Finset.sum_comm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.dotProduct_assoc := dotProduct_assoc


theorem dotProduct_comm [AddCommMonoid α] [CommSemigroup α] (v w : m → α) : v ⬝ᵥ w = w ⬝ᵥ v := by
  /-
    m : Type u_2
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : AddCommMonoid α
    inst✝ : CommSemigroup α
    v w : m → α
    ⊢ Eq (dotProduct v w) (dotProduct w v)
  -/
  simp_rw [dotProduct, mul_comm]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.dotProduct_comm := dotProduct_comm


@[simp]
theorem dotProduct_pUnit [AddCommMonoid α] [Mul α] (v w : PUnit → α) : v ⬝ᵥ w = v ⟨⟩ * w ⟨⟩ := by
  /-
    α : Type v
    inst✝¹ : AddCommMonoid α
    inst✝ : Mul α
    v w : PUnit.{u_10 + 1} → α
    ⊢ Eq (dotProduct v w) (HMul.hMul (v PUnit.unit) (w PUnit.unit))
  -/
  simp [dotProduct]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.dotProduct_pUnit := dotProduct_pUnit


                                                             /-
                                                               n : Type u_3
                                                               α : Type v
                                                               inst✝² : Fintype n
                                                               inst✝¹ : MulOneClass α
                                                               inst✝ : AddCommMonoid α
                                                               v : n → α
                                                               ⊢ Eq (dotProduct v 1) (Finset.univ.sum fun i => v i)
                                                             -/
theorem dotProduct_one (v : n → α) : v ⬝ᵥ 1 = ∑ i, v i := by simp [(· ⬝ᵥ ·)]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.dotProduct_one := dotProduct_one


                                                             /-
                                                               n : Type u_3
                                                               α : Type v
                                                               inst✝² : Fintype n
                                                               inst✝¹ : MulOneClass α
                                                               inst✝ : AddCommMonoid α
                                                               v : n → α
                                                               ⊢ Eq (dotProduct 1 v) (Finset.univ.sum fun i => v i)
                                                             -/
theorem one_dotProduct (v : n → α) : 1 ⬝ᵥ v = ∑ i, v i := by simp [(· ⬝ᵥ ·)]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.one_dotProduct := one_dotProduct


@[simp]
                                           /-
                                             m : Type u_2
                                             α : Type v
                                             inst✝¹ : Fintype m
                                             inst✝ : NonUnitalNonAssocSemiring α
                                             v : m → α
                                             ⊢ Eq (dotProduct v 0) 0
                                           -/
theorem dotProduct_zero : v ⬝ᵥ 0 = 0 := by simp [dotProduct]
                                           /-
                                             🎉 no goals
                                           -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.dotProduct_zero := dotProduct_zero


@[simp]
theorem dotProduct_zero' : (v ⬝ᵥ fun _ => 0) = 0 :=
  dotProduct_zero v


@[deprecated (since := "2024-12-12")] protected alias Matrix.dotProduct_zero' := dotProduct_zero'


@[simp]
                                           /-
                                             m : Type u_2
                                             α : Type v
                                             inst✝¹ : Fintype m
                                             inst✝ : NonUnitalNonAssocSemiring α
                                             v : m → α
                                             ⊢ Eq (dotProduct 0 v) 0
                                           -/
theorem zero_dotProduct : 0 ⬝ᵥ v = 0 := by simp [dotProduct]
                                           /-
                                             🎉 no goals
                                           -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.zero_dotProduct := zero_dotProduct


@[simp]
theorem zero_dotProduct' : (fun _ => (0 : α)) ⬝ᵥ v = 0 :=
  zero_dotProduct v


@[deprecated (since := "2024-12-12")] protected alias Matrix.zero_dotProduct' := zero_dotProduct'


@[simp]
theorem add_dotProduct : (u + v) ⬝ᵥ w = u ⬝ᵥ w + v ⬝ᵥ w := by
  /-
    m : Type u_2
    α : Type v
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    u v w : m → α
    ⊢ Eq (dotProduct (HAdd.hAdd u v) w) (HAdd.hAdd (dotProduct u w) (dotProduct v  …
  -/
  simp [dotProduct, add_mul, Finset.sum_add_distrib]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.add_dotProduct := add_dotProduct


@[simp]
theorem dotProduct_add : u ⬝ᵥ (v + w) = u ⬝ᵥ v + u ⬝ᵥ w := by
  /-
    m : Type u_2
    α : Type v
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocSemiring α
    u v w : m → α
    ⊢ Eq (dotProduct u (HAdd.hAdd v w)) (HAdd.hAdd (dotProduct u v) (dotProduct u  …
  -/
  simp [dotProduct, mul_add, Finset.sum_add_distrib]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.dotProduct_add := dotProduct_add


@[simp]
theorem sum_elim_dotProduct_sum_elim : Sum.elim u x ⬝ᵥ Sum.elim v y = u ⬝ᵥ v + x ⬝ᵥ y := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : NonUnitalNonAssocSemiring α
    u v : m → α
    x y : n → α
    ⊢ Eq (dotProduct (Sum.elim u x) (Sum.elim v y)) (HAdd.hAdd (dotProduct u v) (d …
  -/
  simp [dotProduct]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-12")]
protected alias Matrix.sum_elim_dotProduct_sum_elim := sum_elim_dotProduct_sum_elim


/-- Permuting a vector on the left of a dot product can be transferred to the right. -/
@[simp]
theorem comp_equiv_symm_dotProduct (e : m ≃ n) : u ∘ e.symm ⬝ᵥ x = u ⬝ᵥ x ∘ e :=
  (e.sum_comp _).symm.trans <|
                                       /-
                                         m : Type u_2
                                         n : Type u_3
                                         α : Type v
                                         inst✝² : Fintype m
                                         inst✝¹ : Fintype n
                                         inst✝ : NonUnitalNonAssocSemiring α
                                         u : m → α
                                         x : n → α
                                         e : Equiv m n
                                         x✝¹ : m
                                         x✝ : Membership.mem Finset.univ x✝¹
                                         ⊢ Eq (HMul.hMul (Function.comp u (⇑e.symm) (e x✝¹)) (x (e x✝¹))) (HMul.hMul (u …
                                       -/
    Finset.sum_congr rfl fun _ _ => by simp only [Function.comp, Equiv.symm_apply_apply]
                                       /-
                                         🎉 no goals
                                       -/


@[deprecated (since := "2024-12-12")]
protected alias Matrix.comp_equiv_symm_dotProduct := comp_equiv_symm_dotProduct


/-- Permuting a vector on the right of a dot product can be transferred to the left. -/
@[simp]
theorem dotProduct_comp_equiv_symm (e : n ≃ m) : u ⬝ᵥ x ∘ e.symm = u ∘ e ⬝ᵥ x := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : NonUnitalNonAssocSemiring α
    u : m → α
    x : n → α
    e : Equiv n m
    ⊢ Eq (dotProduct u (Function.comp x ⇑e.symm)) (dotProduct (Function.comp u ⇑e) …
  -/
  simpa only [Equiv.symm_symm] using (comp_equiv_symm_dotProduct u x e.symm).symm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-12")]
protected alias Matrix.dotProduct_comp_equiv_symm := dotProduct_comp_equiv_symm


/-- Permuting vectors on both sides of a dot product is a no-op. -/
@[simp]
theorem comp_equiv_dotProduct_comp_equiv (e : m ≃ n) : x ∘ e ⬝ᵥ y ∘ e = x ⬝ᵥ y := by
  -- Porting note: was `simp only` with all three lemmas
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : Fintype n
    inst✝ : NonUnitalNonAssocSemiring α
    x y : n → α
    e : Equiv m n
    ⊢ Eq (dotProduct (Function.comp x ⇑e) (Function.comp y ⇑e)) (dotProduct x y)
  -/
  rw [← dotProduct_comp_equiv_symm]; simp only [Function.comp_def, Equiv.apply_symm_apply]
                                     /-
                                       🎉 no goals
                                     -/


@[deprecated (since := "2024-12-12")]
protected alias Matrix.comp_equiv_dotProduct_comp_equiv := comp_equiv_dotProduct_comp_equiv


@[simp]
theorem diagonal_dotProduct (i : m) : diagonal v i ⬝ᵥ w = v i * w i := by
  have : ∀ j ≠ i, diagonal v i j * w j = 0 := fun j hij => by
    simp [diagonal_apply_ne' _ hij]
  /-
    m : Type u_2
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : DecidableEq m
    inst✝ : NonUnitalNonAssocSemiring α
    v w : m → α
    i : m
    this : ∀ (j : m), Ne j i → Eq (HMul.hMul (Matrix.diagonal v i j) (w j)) 0
    ⊢ Eq (dotProduct (Matrix.diagonal v i) w) (HMul.hMul (v i) (w i))
  -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  convert Finset.sum_eq_single i (fun j _ => this j) _ using 1 <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[deprecated (since := "2024-12-12")]
protected alias Matrix.diagonal_dotProduct := diagonal_dotProduct


@[simp]
theorem dotProduct_diagonal (i : m) : v ⬝ᵥ diagonal w i = v i * w i := by
  have : ∀ j ≠ i, v j * diagonal w i j = 0 := fun j hij => by
    simp [diagonal_apply_ne' _ hij]
  /-
    m : Type u_2
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : DecidableEq m
    inst✝ : NonUnitalNonAssocSemiring α
    v w : m → α
    i : m
    this : ∀ (j : m), Ne j i → Eq (HMul.hMul (v j) (Matrix.diagonal w i j)) 0
    ⊢ Eq (dotProduct v (Matrix.diagonal w i)) (HMul.hMul (v i) (w i))
  -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  convert Finset.sum_eq_single i (fun j _ => this j) _ using 1 <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[deprecated (since := "2024-12-12")]
protected alias Matrix.dotProduct_diagonal := dotProduct_diagonal


@[simp]
theorem dotProduct_diagonal' (i : m) : (v ⬝ᵥ fun j => diagonal w j i) = v i * w i := by
  have : ∀ j ≠ i, v j * diagonal w j i = 0 := fun j hij => by
    simp [diagonal_apply_ne _ hij]
  /-
    m : Type u_2
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : DecidableEq m
    inst✝ : NonUnitalNonAssocSemiring α
    v w : m → α
    i : m
    this : ∀ (j : m), Ne j i → Eq (HMul.hMul (v j) (Matrix.diagonal w j i)) 0
    ⊢ Eq (dotProduct v fun j => Matrix.diagonal w j i) (HMul.hMul (v i) (w i))
  -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  convert Finset.sum_eq_single i (fun j _ => this j) _ using 1 <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[deprecated (since := "2024-12-12")]
protected alias Matrix.dotProduct_diagonal' := dotProduct_diagonal'


@[simp]
theorem single_dotProduct (x : α) (i : m) : Pi.single i x ⬝ᵥ v = x * v i := by
  -- Porting note: (implicit arg) added `(f := fun _ => α)`
  have : ∀ j ≠ i, Pi.single (f := fun _ => α) i x j * v j = 0 := fun j hij => by
    simp [Pi.single_eq_of_ne hij]
  /-
    m : Type u_2
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : DecidableEq m
    inst✝ : NonUnitalNonAssocSemiring α
    v : m → α
    x : α
    i : m
    this : ∀ (j : m), Ne j i → Eq (HMul.hMul (Pi.single i x j) (v j)) 0
    ⊢ Eq (dotProduct (Pi.single i x) v) (HMul.hMul x (v i))
  -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  convert Finset.sum_eq_single i (fun j _ => this j) _ using 1 <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.single_dotProduct := single_dotProduct


@[simp]
theorem dotProduct_single (x : α) (i : m) : v ⬝ᵥ Pi.single i x = v i * x := by
  -- Porting note: (implicit arg) added `(f := fun _ => α)`
  have : ∀ j ≠ i, v j * Pi.single (f := fun _ => α) i x j = 0 := fun j hij => by
    simp [Pi.single_eq_of_ne hij]
  /-
    m : Type u_2
    α : Type v
    inst✝² : Fintype m
    inst✝¹ : DecidableEq m
    inst✝ : NonUnitalNonAssocSemiring α
    v : m → α
    x : α
    i : m
    this : ∀ (j : m), Ne j i → Eq (HMul.hMul (v j) (Pi.single i x j)) 0
    ⊢ Eq (dotProduct v (Pi.single i x)) (HMul.hMul (v i) x)
  -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  convert Finset.sum_eq_single i (fun j _ => this j) _ using 1 <;> simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.dotProduct_single := dotProduct_single


@[simp]
theorem one_dotProduct_one : (1 : n → α) ⬝ᵥ 1 = Fintype.card n := by
  /-
    n : Type u_3
    α : Type v
    inst✝¹ : Fintype n
    inst✝ : NonAssocSemiring α
    ⊢ Eq (dotProduct 1 1) ↑(Fintype.card n)
  -/
  simp [dotProduct]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-12")]
protected alias Matrix.one_dotProduct_one := one_dotProduct_one


theorem dotProduct_single_one [DecidableEq n] (v : n → α) (i : n) :
    dotProduct v (Pi.single i 1) = v i := by
  /-
    n : Type u_3
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : NonAssocSemiring α
    inst✝ : DecidableEq n
    v : n → α
    i : n
    ⊢ Eq (dotProduct v (Pi.single i 1)) (v i)
  -/
  rw [dotProduct_single, mul_one]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-12")]
protected alias Matrix.dotProduct_single_one := dotProduct_single_one


theorem single_one_dotProduct [DecidableEq n] (i : n) (v : n → α) :
    dotProduct (Pi.single i 1) v = v i := by
  /-
    n : Type u_3
    α : Type v
    inst✝² : Fintype n
    inst✝¹ : NonAssocSemiring α
    inst✝ : DecidableEq n
    i : n
    v : n → α
    ⊢ Eq (dotProduct (Pi.single i 1) v) (v i)
  -/
  rw [single_dotProduct, one_mul]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-12")]
protected alias Matrix.single_one_dotProduct := single_one_dotProduct


@[simp]
                                                   /-
                                                     m : Type u_2
                                                     α : Type v
                                                     inst✝¹ : Fintype m
                                                     inst✝ : NonUnitalNonAssocRing α
                                                     v w : m → α
                                                     ⊢ Eq (dotProduct (Neg.neg v) w) (Neg.neg (dotProduct v w))
                                                   -/
theorem neg_dotProduct : -v ⬝ᵥ w = -(v ⬝ᵥ w) := by simp [dotProduct]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.neg_dotProduct := neg_dotProduct


@[simp]
                                                   /-
                                                     m : Type u_2
                                                     α : Type v
                                                     inst✝¹ : Fintype m
                                                     inst✝ : NonUnitalNonAssocRing α
                                                     v w : m → α
                                                     ⊢ Eq (dotProduct v (Neg.neg w)) (Neg.neg (dotProduct v w))
                                                   -/
theorem dotProduct_neg : v ⬝ᵥ -w = -(v ⬝ᵥ w) := by simp [dotProduct]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.dotProduct_neg := dotProduct_neg


lemma neg_dotProduct_neg : -v ⬝ᵥ -w = v ⬝ᵥ w := by
  /-
    m : Type u_2
    α : Type v
    inst✝¹ : Fintype m
    inst✝ : NonUnitalNonAssocRing α
    v w : m → α
    ⊢ Eq (dotProduct (Neg.neg v) (Neg.neg w)) (dotProduct v w)
  -/
  rw [neg_dotProduct, dotProduct_neg, neg_neg]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-12")]
protected alias Matrix.neg_dotProduct_neg := neg_dotProduct_neg


@[simp]
                                                              /-
                                                                m : Type u_2
                                                                α : Type v
                                                                inst✝¹ : Fintype m
                                                                inst✝ : NonUnitalNonAssocRing α
                                                                u v w : m → α
                                                                ⊢ Eq (dotProduct (HSub.hSub u v) w) (HSub.hSub (dotProduct u w) (dotProduct v  …
                                                              -/
theorem sub_dotProduct : (u - v) ⬝ᵥ w = u ⬝ᵥ w - v ⬝ᵥ w := by simp [sub_eq_add_neg]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.sub_dotProduct := sub_dotProduct


@[simp]
                                                              /-
                                                                m : Type u_2
                                                                α : Type v
                                                                inst✝¹ : Fintype m
                                                                inst✝ : NonUnitalNonAssocRing α
                                                                u v w : m → α
                                                                ⊢ Eq (dotProduct u (HSub.hSub v w)) (HSub.hSub (dotProduct u v) (dotProduct u  …
                                                              -/
theorem dotProduct_sub : u ⬝ᵥ (v - w) = u ⬝ᵥ v - u ⬝ᵥ w := by simp [sub_eq_add_neg]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.dotProduct_sub := dotProduct_sub


@[simp]
theorem smul_dotProduct [IsScalarTower R α α] (x : R) (v w : m → α) :
                                    /-
                                      m : Type u_2
                                      R : Type u_7
                                      α : Type v
                                      inst✝⁵ : Fintype m
                                      inst✝⁴ : Monoid R
                                      inst✝³ : Mul α
                                      inst✝² : AddCommMonoid α
                                      inst✝¹ : DistribMulAction R α
                                      inst✝ : IsScalarTower R α α
                                      x : R
                                      v w : m → α
                                      ⊢ Eq (dotProduct (HSMul.hSMul x v) w) (HSMul.hSMul x (dotProduct v w))
                                    -/
    x • v ⬝ᵥ w = x • (v ⬝ᵥ w) := by simp [dotProduct, Finset.smul_sum, smul_mul_assoc]
                                    /-
                                      🎉 no goals
                                    -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.smul_dotProduct := smul_dotProduct


@[simp]
theorem dotProduct_smul [SMulCommClass R α α] (x : R) (v w : m → α) :
                                    /-
                                      m : Type u_2
                                      R : Type u_7
                                      α : Type v
                                      inst✝⁵ : Fintype m
                                      inst✝⁴ : Monoid R
                                      inst✝³ : Mul α
                                      inst✝² : AddCommMonoid α
                                      inst✝¹ : DistribMulAction R α
                                      inst✝ : SMulCommClass R α α
                                      x : R
                                      v w : m → α
                                      ⊢ Eq (dotProduct v (HSMul.hSMul x w)) (HSMul.hSMul x (dotProduct v w))
                                    -/
    v ⬝ᵥ x • w = x • (v ⬝ᵥ w) := by simp [dotProduct, Finset.smul_sum, mul_smul_comm]
                                    /-
                                      🎉 no goals
                                    -/


@[deprecated (since := "2024-12-12")] protected alias Matrix.dotProduct_smul := dotProduct_smul


/-- `M * N` is the usual product of matrices `M` and `N`, i.e. we have that
`(M * N) i k` is the dot product of the `i`-th row of `M` by the `k`-th column of `N`.
This is currently only defined when `m` is finite. -/
-- We want to be lower priority than `instHMul`, but without this we can't have operands with
-- implicit dimensions.
@[default_instance 100]
instance [Fintype m] [Mul α] [AddCommMonoid α] :
    HMul (Matrix l m α) (Matrix m n α) (Matrix l n α) where
  hMul M N := fun i k => (fun j => M i j) ⬝ᵥ fun j => N j k


theorem mul_apply [Fintype m] [Mul α] [AddCommMonoid α] {M : Matrix l m α} {N : Matrix m n α}
    {i k} : (M * N) i k = ∑ j, M i j * N j k :=
  rfl


instance [Fintype n] [Mul α] [AddCommMonoid α] : Mul (Matrix n n α) where mul M N := M * N


theorem mul_apply' [Fintype m] [Mul α] [AddCommMonoid α] {M : Matrix l m α} {N : Matrix m n α}
    {i k} : (M * N) i k = (fun j => M i j) ⬝ᵥ fun j => N j k :=
  rfl


theorem two_mul_expl {R : Type*} [CommRing R] (A B : Matrix (Fin 2) (Fin 2) R) :
    (A * B) 0 0 = A 0 0 * B 0 0 + A 0 1 * B 1 0 ∧
    (A * B) 0 1 = A 0 0 * B 0 1 + A 0 1 * B 1 1 ∧
    (A * B) 1 0 = A 1 0 * B 0 0 + A 1 1 * B 1 0 ∧
    (A * B) 1 1 = A 1 0 * B 0 1 + A 1 1 * B 1 1 := by
  /-
    R : Type u_10
    inst✝ : CommRing R
    A B : Matrix (Fin 2) (Fin 2) R
    ⊢ And (Eq (HMul.hMul A B 0 0) (HAdd.hAdd (HMul.hMul (A 0 0) (B 0 0)) (HMul.hMu …
  -/
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    /-
      case refine_1
      R : Type u_10
      inst✝ : CommRing R
      A B : Matrix (Fin 2) (Fin 2) R
      ⊢ Eq (HMul.hMul A B 0 0) (HAdd.hAdd (HMul.hMul (A 0 0) (B 0 0)) (HMul.hMul (A  …
    -/
    /-
      case refine_1
      R : Type u_10
      inst✝ : CommRing R
      A B : Matrix (Fin 2) (Fin 2) R
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd ((Finset.range 0).sum fun x => dite (LT.lt x 2) (fu …
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
      case refine_4
      R : Type u_10
      inst✝ : CommRing R
      A B : Matrix (Fin 2) (Fin 2) R
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd ((Finset.range 0).sum fun x => dite (LT.lt x 2) (fu …
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem smul_mul [Fintype n] [Monoid R] [DistribMulAction R α] [IsScalarTower R α α] (a : R)
    (M : Matrix m n α) (N : Matrix n l α) : (a • M) * N = a • (M * N) := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    R : Type u_7
    α : Type v
    inst✝⁵ : AddCommMonoid α
    inst✝⁴ : Mul α
    inst✝³ : Fintype n
    inst✝² : Monoid R
    inst✝¹ : DistribMulAction R α
    inst✝ : IsScalarTower R α α
    a : R
    M : Matrix m n α
    N : Matrix n l α
    ⊢ Eq (HMul.hMul (HSMul.hSMul a M) N) (HSMul.hSMul a (HMul.hMul M N))
  -/
  ext
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    R : Type u_7
    α : Type v
    inst✝⁵ : AddCommMonoid α
    inst✝⁴ : Mul α
    inst✝³ : Fintype n
    inst✝² : Monoid R
    inst✝¹ : DistribMulAction R α
    inst✝ : IsScalarTower R α α
    a : R
    M : Matrix m n α
    N : Matrix n l α
    i✝ : m
    j✝ : l
    ⊢ Eq (HMul.hMul (HSMul.hSMul a M) N i✝ j✝) (HSMul.hSMul a (HMul.hMul M N) i✝ j✝)
  -/
  apply smul_dotProduct a
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_smul [Fintype n] [Monoid R] [DistribMulAction R α] [SMulCommClass R α α]
    (M : Matrix m n α) (a : R) (N : Matrix n l α) : M * (a • N) = a • (M * N) := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    R : Type u_7
    α : Type v
    inst✝⁵ : AddCommMonoid α
    inst✝⁴ : Mul α
    inst✝³ : Fintype n
    inst✝² : Monoid R
    inst✝¹ : DistribMulAction R α
    inst✝ : SMulCommClass R α α
    M : Matrix m n α
    a : R
    N : Matrix n l α
    ⊢ Eq (HMul.hMul M (HSMul.hSMul a N)) (HSMul.hSMul a (HMul.hMul M N))
  -/
  ext
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    R : Type u_7
    α : Type v
    inst✝⁵ : AddCommMonoid α
    inst✝⁴ : Mul α
    inst✝³ : Fintype n
    inst✝² : Monoid R
    inst✝¹ : DistribMulAction R α
    inst✝ : SMulCommClass R α α
    M : Matrix m n α
    a : R
    N : Matrix n l α
    i✝ : m
    j✝ : l
    ⊢ Eq (HMul.hMul M (HSMul.hSMul a N) i✝ j✝) (HSMul.hSMul a (HMul.hMul M N) i✝ j✝)
  -/
  apply dotProduct_smul
  /-
    🎉 no goals
  -/


@[simp]
protected theorem mul_zero [Fintype n] (M : Matrix m n α) : M * (0 : Matrix n o α) = 0 := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n
    M : Matrix m n α
    ⊢ Eq (HMul.hMul M 0) 0
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n
    M : Matrix m n α
    i✝ : m
    j✝ : o
    ⊢ Eq (HMul.hMul M 0 i✝ j✝) (0 i✝ j✝)
  -/
  apply dotProduct_zero
  /-
    🎉 no goals
  -/


@[simp]
protected theorem zero_mul [Fintype m] (M : Matrix m n α) : (0 : Matrix l m α) * M = 0 := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    M : Matrix m n α
    ⊢ Eq (HMul.hMul 0 M) 0
  -/
  ext
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    M : Matrix m n α
    i✝ : l
    j✝ : n
    ⊢ Eq (HMul.hMul 0 M i✝ j✝) (0 i✝ j✝)
  -/
  apply zero_dotProduct
  /-
    🎉 no goals
  -/


protected theorem mul_add [Fintype n] (L : Matrix m n α) (M N : Matrix n o α) :
    L * (M + N) = L * M + L * N := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n
    L : Matrix m n α
    M N : Matrix n o α
    ⊢ Eq (HMul.hMul L (HAdd.hAdd M N)) (HAdd.hAdd (HMul.hMul L M) (HMul.hMul L N))
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n
    L : Matrix m n α
    M N : Matrix n o α
    i✝ : m
    j✝ : o
    ⊢ Eq (HMul.hMul L (HAdd.hAdd M N) i✝ j✝) (HAdd.hAdd (HMul.hMul L M) (HMul.hMul …
  -/
  apply dotProduct_add
  /-
    🎉 no goals
  -/


protected theorem add_mul [Fintype m] (L M : Matrix l m α) (N : Matrix m n α) :
    (L + M) * N = L * N + M * N := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    L M : Matrix l m α
    N : Matrix m n α
    ⊢ Eq (HMul.hMul (HAdd.hAdd L M) N) (HAdd.hAdd (HMul.hMul L N) (HMul.hMul M N))
  -/
  ext
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    L M : Matrix l m α
    N : Matrix m n α
    i✝ : l
    j✝ : n
    ⊢ Eq (HMul.hMul (HAdd.hAdd L M) N i✝ j✝) (HAdd.hAdd (HMul.hMul L N) (HMul.hMul …
  -/
  apply add_dotProduct
  /-
    🎉 no goals
  -/


instance nonUnitalNonAssocSemiring [Fintype n] : NonUnitalNonAssocSemiring (Matrix n n α) :=
  { Matrix.addCommMonoid with
    mul_zero := Matrix.mul_zero
    zero_mul := Matrix.zero_mul
    left_distrib := Matrix.mul_add
    right_distrib := Matrix.add_mul }


@[simp]
theorem diagonal_mul [Fintype m] [DecidableEq m] (d : m → α) (M : Matrix m n α) (i j) :
    (diagonal d * M) i j = d i * M i j :=
  diagonal_dotProduct _ _ _


@[simp]
theorem mul_diagonal [Fintype n] [DecidableEq n] (d : n → α) (M : Matrix m n α) (i j) :
    (M * diagonal d) i j = M i j * d j := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    d : n → α
    M : Matrix m n α
    i : m
    j : n
    ⊢ Eq (HMul.hMul M (Matrix.diagonal d) i j) (HMul.hMul (M i j) (d j))
  -/
  rw [← diagonal_transpose]
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    d : n → α
    M : Matrix m n α
    i : m
    j : n
    ⊢ Eq (HMul.hMul M (Matrix.diagonal d).transpose i j) (HMul.hMul (M i j) (d j))
  -/
  apply dotProduct_diagonal
  /-
    🎉 no goals
  -/


@[simp]
theorem diagonal_mul_diagonal [Fintype n] [DecidableEq n] (d₁ d₂ : n → α) :
    diagonal d₁ * diagonal d₂ = diagonal fun i => d₁ i * d₂ i := by
  /-
    n : Type u_3
    α : Type v
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    d₁ d₂ : n → α
    ⊢ Eq (HMul.hMul (Matrix.diagonal d₁) (Matrix.diagonal d₂)) (Matrix.diagonal fu …
  -/
  ext i j
  /-
    case a
    n : Type u_3
    α : Type v
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    d₁ d₂ : n → α
    i j : n
    ⊢ Eq (HMul.hMul (Matrix.diagonal d₁) (Matrix.diagonal d₂) i j) (Matrix.diagona …
  -/
  by_cases h : i = j <;>
  /-
    case pos
    n : Type u_3
    α : Type v
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    d₁ d₂ : n → α
    i j : n
    h : Eq i j
    ⊢ Eq (HMul.hMul (Matrix.diagonal d₁) (Matrix.diagonal d₂) i j) (Matrix.diagona …
  -/
  /-
    🎉 no goals
  -/
  simp [h]
  /-
    🎉 no goals
  -/


theorem diagonal_mul_diagonal' [Fintype n] [DecidableEq n] (d₁ d₂ : n → α) :
    diagonal d₁ * diagonal d₂ = diagonal fun i => d₁ i * d₂ i :=
  diagonal_mul_diagonal _ _


theorem smul_eq_diagonal_mul [Fintype m] [DecidableEq m] (M : Matrix m n α) (a : α) :
    a • M = (diagonal fun _ => a) * M := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    M : Matrix m n α
    a : α
    ⊢ Eq (HSMul.hSMul a M) (HMul.hMul (Matrix.diagonal fun x => a) M)
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    M : Matrix m n α
    a : α
    i✝ : m
    j✝ : n
    ⊢ Eq (HSMul.hSMul a M i✝ j✝) (HMul.hMul (Matrix.diagonal fun x => a) M i✝ j✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem op_smul_eq_mul_diagonal [Fintype n] [DecidableEq n] (M : Matrix m n α) (a : α) :
    MulOpposite.op a • M = M * (diagonal fun _ : n => a) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix m n α
    a : α
    ⊢ Eq (HSMul.hSMul (MulOpposite.op a) M) (HMul.hMul M (Matrix.diagonal fun x => …
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : NonUnitalNonAssocSemiring α
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix m n α
    a : α
    i✝ : m
    j✝ : n
    ⊢ Eq (HSMul.hSMul (MulOpposite.op a) M i✝ j✝) (HMul.hMul M (Matrix.diagonal fu …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Left multiplication by a matrix, as an `AddMonoidHom` from matrices to matrices. -/
@[simps]
def addMonoidHomMulLeft [Fintype m] (M : Matrix l m α) : Matrix m n α →+ Matrix l n α where
  toFun x := M * x
  map_zero' := Matrix.mul_zero _
  map_add' := Matrix.mul_add _


/-- Right multiplication by a matrix, as an `AddMonoidHom` from matrices to matrices. -/
@[simps]
def addMonoidHomMulRight [Fintype m] (M : Matrix m n α) : Matrix l m α →+ Matrix l n α where
  toFun x := x * M
  map_zero' := Matrix.zero_mul _
  map_add' _ _ := Matrix.add_mul _ _ _


protected theorem sum_mul [Fintype m] (s : Finset β) (f : β → Matrix l m α) (M : Matrix m n α) :
    (∑ a ∈ s, f a) * M = ∑ a ∈ s, f a * M :=
  map_sum (addMonoidHomMulRight M) f s


protected theorem mul_sum [Fintype m] (s : Finset β) (f : β → Matrix m n α) (M : Matrix l m α) :
    (M * ∑ a ∈ s, f a) = ∑ a ∈ s, M * f a :=
  map_sum (addMonoidHomMulLeft M) f s


/-- This instance enables use with `smul_mul_assoc`. -/
instance Semiring.isScalarTower [Fintype n] [Monoid R] [DistribMulAction R α]
    [IsScalarTower R α α] : IsScalarTower R (Matrix n n α) (Matrix n n α) :=
  ⟨fun r m n => Matrix.smul_mul r m n⟩


/-- This instance enables use with `mul_smul_comm`. -/
instance Semiring.smulCommClass [Fintype n] [Monoid R] [DistribMulAction R α]
    [SMulCommClass R α α] : SMulCommClass R (Matrix n n α) (Matrix n n α) :=
  ⟨fun r m n => (Matrix.mul_smul m r n).symm⟩


@[simp]
protected theorem one_mul [Fintype m] [DecidableEq m] (M : Matrix m n α) :
    (1 : Matrix m m α) * M = M := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : NonAssocSemiring α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    M : Matrix m n α
    ⊢ Eq (HMul.hMul 1 M) M
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : NonAssocSemiring α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    M : Matrix m n α
    i✝ : m
    j✝ : n
    ⊢ Eq (HMul.hMul 1 M i✝ j✝) (M i✝ j✝)
  -/
  rw [← diagonal_one, diagonal_mul, one_mul]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem mul_one [Fintype n] [DecidableEq n] (M : Matrix m n α) :
    M * (1 : Matrix n n α) = M := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : NonAssocSemiring α
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix m n α
    ⊢ Eq (HMul.hMul M 1) M
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : NonAssocSemiring α
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix m n α
    i✝ : m
    j✝ : n
    ⊢ Eq (HMul.hMul M 1 i✝ j✝) (M i✝ j✝)
  -/
  rw [← diagonal_one, mul_diagonal, mul_one]
  /-
    🎉 no goals
  -/


instance nonAssocSemiring [Fintype n] [DecidableEq n] : NonAssocSemiring (Matrix n n α) :=
  { Matrix.nonUnitalNonAssocSemiring, Matrix.instAddCommMonoidWithOne with
    one := 1
    one_mul := Matrix.one_mul
    mul_one := Matrix.mul_one }


@[simp]
theorem map_mul [Fintype n] {L : Matrix m n α} {M : Matrix n o α} [NonAssocSemiring β]
    {f : α →+* β} : (L * M).map f = L.map f * M.map f := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    β : Type w
    inst✝² : NonAssocSemiring α
    inst✝¹ : Fintype n
    L : Matrix m n α
    M : Matrix n o α
    inst✝ : NonAssocSemiring β
    f : RingHom α β
    ⊢ Eq ((HMul.hMul L M).map ⇑f) (HMul.hMul (L.map ⇑f) (M.map ⇑f))
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    β : Type w
    inst✝² : NonAssocSemiring α
    inst✝¹ : Fintype n
    L : Matrix m n α
    M : Matrix n o α
    inst✝ : NonAssocSemiring β
    f : RingHom α β
    i✝ : m
    j✝ : o
    ⊢ Eq ((HMul.hMul L M).map (⇑f) i✝ j✝) (HMul.hMul (L.map ⇑f) (M.map ⇑f) i✝ j✝)
  -/
  simp [mul_apply, map_sum]
  /-
    🎉 no goals
  -/


theorem smul_one_eq_diagonal [DecidableEq m] (a : α) :
    a • (1 : Matrix m m α) = diagonal fun _ => a := by
  /-
    m : Type u_2
    α : Type v
    inst✝¹ : NonAssocSemiring α
    inst✝ : DecidableEq m
    a : α
    ⊢ Eq (HSMul.hSMul a 1) (Matrix.diagonal fun x => a)
  -/
  simp_rw [← diagonal_one, ← diagonal_smul, Pi.smul_def, smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


theorem op_smul_one_eq_diagonal [DecidableEq m] (a : α) :
    MulOpposite.op a • (1 : Matrix m m α) = diagonal fun _ => a := by
  /-
    m : Type u_2
    α : Type v
    inst✝¹ : NonAssocSemiring α
    inst✝ : DecidableEq m
    a : α
    ⊢ Eq (HSMul.hSMul (MulOpposite.op a) 1) (Matrix.diagonal fun x => a)
  -/
  simp_rw [← diagonal_one, ← diagonal_smul, Pi.smul_def, op_smul_eq_mul, one_mul]
  /-
    🎉 no goals
  -/


protected theorem mul_assoc (L : Matrix l m α) (M : Matrix m n α) (N : Matrix n o α) :
    L * M * N = L * (M * N) := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝² : NonUnitalSemiring α
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    L : Matrix l m α
    M : Matrix m n α
    N : Matrix n o α
    ⊢ Eq (HMul.hMul (HMul.hMul L M) N) (HMul.hMul L (HMul.hMul M N))
  -/
  ext
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝² : NonUnitalSemiring α
    inst✝¹ : Fintype m
    inst✝ : Fintype n
    L : Matrix l m α
    M : Matrix m n α
    N : Matrix n o α
    i✝ : l
    j✝ : o
    ⊢ Eq (HMul.hMul (HMul.hMul L M) N i✝ j✝) (HMul.hMul L (HMul.hMul M N) i✝ j✝)
  -/
  apply dotProduct_assoc
  /-
    🎉 no goals
  -/


instance nonUnitalSemiring : NonUnitalSemiring (Matrix n n α) :=
  { Matrix.nonUnitalNonAssocSemiring with mul_assoc := Matrix.mul_assoc }


instance semiring [Fintype n] [DecidableEq n] : Semiring (Matrix n n α) :=
  { Matrix.nonUnitalSemiring, Matrix.nonAssocSemiring with }


@[simp]
protected theorem neg_mul (M : Matrix m n α) (N : Matrix n o α) : (-M) * N = -(M * N) := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype n
    M : Matrix m n α
    N : Matrix n o α
    ⊢ Eq (HMul.hMul (Neg.neg M) N) (Neg.neg (HMul.hMul M N))
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype n
    M : Matrix m n α
    N : Matrix n o α
    i✝ : m
    j✝ : o
    ⊢ Eq (HMul.hMul (Neg.neg M) N i✝ j✝) (Neg.neg (HMul.hMul M N) i✝ j✝)
  -/
  apply neg_dotProduct
  /-
    🎉 no goals
  -/


@[simp]
protected theorem mul_neg (M : Matrix m n α) (N : Matrix n o α) : M * (-N) = -(M * N) := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype n
    M : Matrix m n α
    N : Matrix n o α
    ⊢ Eq (HMul.hMul M (Neg.neg N)) (Neg.neg (HMul.hMul M N))
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype n
    M : Matrix m n α
    N : Matrix n o α
    i✝ : m
    j✝ : o
    ⊢ Eq (HMul.hMul M (Neg.neg N) i✝ j✝) (Neg.neg (HMul.hMul M N) i✝ j✝)
  -/
  apply dotProduct_neg
  /-
    🎉 no goals
  -/


protected theorem sub_mul (M M' : Matrix m n α) (N : Matrix n o α) :
    (M - M') * N = M * N - M' * N := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype n
    M M' : Matrix m n α
    N : Matrix n o α
    ⊢ Eq (HMul.hMul (HSub.hSub M M') N) (HSub.hSub (HMul.hMul M N) (HMul.hMul M' N))
  -/
  rw [sub_eq_add_neg, Matrix.add_mul, Matrix.neg_mul, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


protected theorem mul_sub (M : Matrix m n α) (N N' : Matrix n o α) :
    M * (N - N') = M * N - M * N' := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype n
    M : Matrix m n α
    N N' : Matrix n o α
    ⊢ Eq (HMul.hMul M (HSub.hSub N N')) (HSub.hSub (HMul.hMul M N) (HMul.hMul M N'))
  -/
  rw [sub_eq_add_neg, Matrix.mul_add, Matrix.mul_neg, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


instance nonUnitalNonAssocRing : NonUnitalNonAssocRing (Matrix n n α) :=
  { Matrix.nonUnitalNonAssocSemiring, Matrix.addCommGroup with }


instance instNonUnitalRing [Fintype n] [NonUnitalRing α] : NonUnitalRing (Matrix n n α) :=
  { Matrix.nonUnitalSemiring, Matrix.addCommGroup with }


instance instNonAssocRing [Fintype n] [DecidableEq n] [NonAssocRing α] :
    NonAssocRing (Matrix n n α) :=
  { Matrix.nonAssocSemiring, Matrix.instAddCommGroupWithOne with }


instance instRing [Fintype n] [DecidableEq n] [Ring α] : Ring (Matrix n n α) :=
  { Matrix.semiring, Matrix.instAddCommGroupWithOne with }


@[simp]
theorem mul_mul_left [Fintype n] (M : Matrix m n α) (N : Matrix n o α) (a : α) :
    (of fun i j => a * M i j) * N = a • (M * N) :=
  smul_mul a M N


theorem smul_eq_mul_diagonal [Fintype n] [DecidableEq n] (M : Matrix m n α) (a : α) :
    a • M = M * diagonal fun _ => a := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : CommSemiring α
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix m n α
    a : α
    ⊢ Eq (HSMul.hSMul a M) (HMul.hMul M (Matrix.diagonal fun x => a))
  -/
  ext
  /-
    case a
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : CommSemiring α
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix m n α
    a : α
    i✝ : m
    j✝ : n
    ⊢ Eq (HSMul.hSMul a M i✝ j✝) (HMul.hMul M (Matrix.diagonal fun x => a) i✝ j✝)
  -/
  simp [mul_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_mul_right [Fintype n] (M : Matrix m n α) (N : Matrix n o α) (a : α) :
    (M * of fun i j => a * N i j) = a • (M * N) :=
  mul_smul M a N


/-- For two vectors `w` and `v`, `vecMulVec w v i j` is defined to be `w i * v j`.
    Put another way, `vecMulVec w v` is exactly `col w * row v`. -/
def vecMulVec [Mul α] (w : m → α) (v : n → α) : Matrix m n α :=
  of fun x y => w x * v y

-- TODO: set as an equation lemma for `vecMulVec`, see https://github.com/leanprover-community/mathlib4/pull/3024

theorem vecMulVec_apply [Mul α] (w : m → α) (v : n → α) (i j) : vecMulVec w v i j = w i * v j :=
  rfl


/--
`M *ᵥ v` (notation for `mulVec M v`) is the matrix-vector product of matrix `M` and vector `v`,
where `v` is seen as a column vector.
Put another way, `M *ᵥ v` is the vector whose entries are those of `M * col v` (see `col_mulVec`).

The notation has precedence 73, which comes immediately before ` ⬝ᵥ ` for `dotProduct`,
so that `A *ᵥ v ⬝ᵥ B *ᵥ w` is parsed as `(A *ᵥ v) ⬝ᵥ (B *ᵥ w)`.
-/
def mulVec [Fintype n] (M : Matrix m n α) (v : n → α) : m → α
  | i => (fun j => M i j) ⬝ᵥ v


@[inherit_doc]
scoped infixr:73 " *ᵥ " => Matrix.mulVec


/--
`v ᵥ* M` (notation for `vecMul v M`) is the vector-matrix product of vector `v` and matrix `M`,
where `v` is seen as a row vector.
Put another way, `v ᵥ* M` is the vector whose entries are those of `row v * M` (see `row_vecMul`).

The notation has precedence 73, which comes immediately before ` ⬝ᵥ ` for `dotProduct`,
so that `v ᵥ* A ⬝ᵥ w ᵥ* B` is parsed as `(v ᵥ* A) ⬝ᵥ (w ᵥ* B)`.
-/
def vecMul [Fintype m] (v : m → α) (M : Matrix m n α) : n → α
  | j => v ⬝ᵥ fun i => M i j


@[inherit_doc]
scoped infixl:73 " ᵥ* " => Matrix.vecMul


/-- Left multiplication by a matrix, as an `AddMonoidHom` from vectors to vectors. -/
@[simps]
def mulVec.addMonoidHomLeft [Fintype n] (v : n → α) : Matrix m n α →+ m → α where
  toFun M := M *ᵥ v
  map_zero' := by
    /-
      l : Type u_1
      m : Type u_2
      n : Type u_3
      o : Type u_4
      m' : o → Type u_5
      n' : o → Type u_6
      R : Type u_7
      S : Type u_8
      α : Type v
      β : Type w
      γ : Type u_9
      inst✝¹ : NonUnitalNonAssocSemiring α
      inst✝ : Fintype n
      v : n → α
      ⊢ Eq ((fun M => M.mulVec v) 0) 0
    -/
    ext
    /-
      case h
      l : Type u_1
      m : Type u_2
      n : Type u_3
      o : Type u_4
      m' : o → Type u_5
      n' : o → Type u_6
      R : Type u_7
      S : Type u_8
      α : Type v
      β : Type w
      γ : Type u_9
      inst✝¹ : NonUnitalNonAssocSemiring α
      inst✝ : Fintype n
      v : n → α
      x✝ : m
      ⊢ Eq ((fun M => M.mulVec v) 0 x✝) (0 x✝)
    -/
    simp [mulVec]
    /-
      🎉 no goals
    -/
  map_add' x y := by
    /-
      l : Type u_1
      m : Type u_2
      n : Type u_3
      o : Type u_4
      m' : o → Type u_5
      n' : o → Type u_6
      R : Type u_7
      S : Type u_8
      α : Type v
      β : Type w
      γ : Type u_9
      inst✝¹ : NonUnitalNonAssocSemiring α
      inst✝ : Fintype n
      v : n → α
      x y : Matrix m n α
      ⊢ Eq ({ toFun := fun M => M.mulVec v, map_zero' := ⋯ }.toFun (HAdd.hAdd x y))  …
    -/
    ext m
    /-
      case h
      l : Type u_1
      m✝ : Type u_2
      n : Type u_3
      o : Type u_4
      m' : o → Type u_5
      n' : o → Type u_6
      R : Type u_7
      S : Type u_8
      α : Type v
      β : Type w
      γ : Type u_9
      inst✝¹ : NonUnitalNonAssocSemiring α
      inst✝ : Fintype n
      v : n → α
      x y : Matrix m✝ n α
      m : m✝
      ⊢ Eq ({ toFun := fun M => M.mulVec v, map_zero' := ⋯ }.toFun (HAdd.hAdd x y) m …
    -/
    apply add_dotProduct
    /-
      🎉 no goals
    -/


/-- The `i`th row of the multiplication is the same as the `vecMul` with the `i`th row of `A`. -/
theorem mul_apply_eq_vecMul [Fintype n] (A : Matrix m n α) (B : Matrix n o α) (i : m) :
    (A * B) i = A i ᵥ* B :=
  rfl


theorem mulVec_diagonal [Fintype m] [DecidableEq m] (v w : m → α) (x : m) :
    (diagonal v *ᵥ w) x = v x * w x :=
  diagonal_dotProduct v w x


theorem vecMul_diagonal [Fintype m] [DecidableEq m] (v w : m → α) (x : m) :
    (v ᵥ* diagonal w) x = v x * w x :=
  dotProduct_diagonal' v w x


/-- Associate the dot product of `mulVec` to the left. -/
theorem dotProduct_mulVec [Fintype n] [Fintype m] [NonUnitalSemiring R] (v : m → R)
    (A : Matrix m n R) (w : n → R) : v ⬝ᵥ A *ᵥ w = v ᵥ* A ⬝ᵥ w := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_7
    inst✝² : Fintype n
    inst✝¹ : Fintype m
    inst✝ : NonUnitalSemiring R
    v : m → R
    A : Matrix m n R
    w : n → R
    ⊢ Eq (dotProduct v (A.mulVec w)) (dotProduct (Matrix.vecMul v A) w)
  -/
  simp only [dotProduct, vecMul, mulVec, Finset.mul_sum, Finset.sum_mul, mul_assoc]
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_7
    inst✝² : Fintype n
    inst✝¹ : Fintype m
    inst✝ : NonUnitalSemiring R
    v : m → R
    A : Matrix m n R
    w : n → R
    ⊢ Eq (Finset.univ.sum fun x => Finset.univ.sum fun i => HMul.hMul (v x) (HMul. …
  -/
  exact Finset.sum_comm
  /-
    🎉 no goals
  -/


@[simp]
theorem mulVec_zero [Fintype n] (A : Matrix m n α) : A *ᵥ 0 = 0 := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n
    A : Matrix m n α
    ⊢ Eq (A.mulVec 0) 0
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n
    A : Matrix m n α
    x✝ : m
    ⊢ Eq (A.mulVec 0 x✝) (0 x✝)
  -/
  simp [mulVec]
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_vecMul [Fintype m] (A : Matrix m n α) : 0 ᵥ* A = 0 := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    A : Matrix m n α
    ⊢ Eq (Matrix.vecMul 0 A) 0
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    A : Matrix m n α
    x✝ : n
    ⊢ Eq (Matrix.vecMul 0 A x✝) (0 x✝)
  -/
  simp [vecMul]
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_mulVec [Fintype n] (v : n → α) : (0 : Matrix m n α) *ᵥ v = 0 := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n
    v : n → α
    ⊢ Eq (Matrix.mulVec 0 v) 0
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n
    v : n → α
    x✝ : m
    ⊢ Eq (Matrix.mulVec 0 v x✝) (0 x✝)
  -/
  simp [mulVec]
  /-
    🎉 no goals
  -/


@[simp]
theorem vecMul_zero [Fintype m] (v : m → α) : v ᵥ* (0 : Matrix m n α) = 0 := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    v : m → α
    ⊢ Eq (Matrix.vecMul v 0) 0
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    v : m → α
    x✝ : n
    ⊢ Eq (Matrix.vecMul v 0 x✝) (0 x✝)
  -/
  simp [vecMul]
  /-
    🎉 no goals
  -/


theorem smul_mulVec_assoc [Fintype n] [Monoid R] [DistribMulAction R α] [IsScalarTower R α α]
    (a : R) (A : Matrix m n α) (b : n → α) : (a • A) *ᵥ b = a • A *ᵥ b := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_7
    α : Type v
    inst✝⁴ : NonUnitalNonAssocSemiring α
    inst✝³ : Fintype n
    inst✝² : Monoid R
    inst✝¹ : DistribMulAction R α
    inst✝ : IsScalarTower R α α
    a : R
    A : Matrix m n α
    b : n → α
    ⊢ Eq ((HSMul.hSMul a A).mulVec b) (HSMul.hSMul a (A.mulVec b))
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    R : Type u_7
    α : Type v
    inst✝⁴ : NonUnitalNonAssocSemiring α
    inst✝³ : Fintype n
    inst✝² : Monoid R
    inst✝¹ : DistribMulAction R α
    inst✝ : IsScalarTower R α α
    a : R
    A : Matrix m n α
    b : n → α
    x✝ : m
    ⊢ Eq ((HSMul.hSMul a A).mulVec b x✝) (HSMul.hSMul a (A.mulVec b) x✝)
  -/
  apply smul_dotProduct
  /-
    🎉 no goals
  -/


theorem mulVec_add [Fintype n] (A : Matrix m n α) (x y : n → α) :
    A *ᵥ (x + y) = A *ᵥ x + A *ᵥ y := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n
    A : Matrix m n α
    x y : n → α
    ⊢ Eq (A.mulVec (HAdd.hAdd x y)) (HAdd.hAdd (A.mulVec x) (A.mulVec y))
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n
    A : Matrix m n α
    x y : n → α
    x✝ : m
    ⊢ Eq (A.mulVec (HAdd.hAdd x y) x✝) (HAdd.hAdd (A.mulVec x) (A.mulVec y) x✝)
  -/
  apply dotProduct_add
  /-
    🎉 no goals
  -/


theorem add_mulVec [Fintype n] (A B : Matrix m n α) (x : n → α) :
    (A + B) *ᵥ x = A *ᵥ x + B *ᵥ x := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n
    A B : Matrix m n α
    x : n → α
    ⊢ Eq ((HAdd.hAdd A B).mulVec x) (HAdd.hAdd (A.mulVec x) (B.mulVec x))
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype n
    A B : Matrix m n α
    x : n → α
    x✝ : m
    ⊢ Eq ((HAdd.hAdd A B).mulVec x x✝) (HAdd.hAdd (A.mulVec x) (B.mulVec x) x✝)
  -/
  apply add_dotProduct
  /-
    🎉 no goals
  -/


theorem vecMul_add [Fintype m] (A B : Matrix m n α) (x : m → α) :
    x ᵥ* (A + B) = x ᵥ* A + x ᵥ* B := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    A B : Matrix m n α
    x : m → α
    ⊢ Eq (Matrix.vecMul x (HAdd.hAdd A B)) (HAdd.hAdd (Matrix.vecMul x A) (Matrix. …
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    A B : Matrix m n α
    x : m → α
    x✝ : n
    ⊢ Eq (Matrix.vecMul x (HAdd.hAdd A B) x✝) (HAdd.hAdd (Matrix.vecMul x A) (Matr …
  -/
  apply dotProduct_add
  /-
    🎉 no goals
  -/


theorem add_vecMul [Fintype m] (A : Matrix m n α) (x y : m → α) :
    (x + y) ᵥ* A = x ᵥ* A + y ᵥ* A := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    A : Matrix m n α
    x y : m → α
    ⊢ Eq (Matrix.vecMul (HAdd.hAdd x y) A) (HAdd.hAdd (Matrix.vecMul x A) (Matrix. …
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocSemiring α
    inst✝ : Fintype m
    A : Matrix m n α
    x y : m → α
    x✝ : n
    ⊢ Eq (Matrix.vecMul (HAdd.hAdd x y) A x✝) (HAdd.hAdd (Matrix.vecMul x A) (Matr …
  -/
  apply add_dotProduct
  /-
    🎉 no goals
  -/


theorem vecMul_smul [Fintype n] [Monoid R] [NonUnitalNonAssocSemiring S] [DistribMulAction R S]
    [IsScalarTower R S S] (M : Matrix n m S) (b : R) (v : n → S) :
    (b • v) ᵥ* M = b • v ᵥ* M := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_7
    S : Type u_8
    inst✝⁴ : Fintype n
    inst✝³ : Monoid R
    inst✝² : NonUnitalNonAssocSemiring S
    inst✝¹ : DistribMulAction R S
    inst✝ : IsScalarTower R S S
    M : Matrix n m S
    b : R
    v : n → S
    ⊢ Eq (Matrix.vecMul (HSMul.hSMul b v) M) (HSMul.hSMul b (Matrix.vecMul v M))
  -/
  ext i
  /-
    case h
    m : Type u_2
    n : Type u_3
    R : Type u_7
    S : Type u_8
    inst✝⁴ : Fintype n
    inst✝³ : Monoid R
    inst✝² : NonUnitalNonAssocSemiring S
    inst✝¹ : DistribMulAction R S
    inst✝ : IsScalarTower R S S
    M : Matrix n m S
    b : R
    v : n → S
    i : m
    ⊢ Eq (Matrix.vecMul (HSMul.hSMul b v) M i) (HSMul.hSMul b (Matrix.vecMul v M) i)
  -/
  simp only [vecMul, dotProduct, Finset.smul_sum, Pi.smul_apply, smul_mul_assoc]
  /-
    🎉 no goals
  -/


theorem mulVec_smul [Fintype n] [Monoid R] [NonUnitalNonAssocSemiring S] [DistribMulAction R S]
    [SMulCommClass R S S] (M : Matrix m n S) (b : R) (v : n → S) :
    M *ᵥ (b • v) = b • M *ᵥ v := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_7
    S : Type u_8
    inst✝⁴ : Fintype n
    inst✝³ : Monoid R
    inst✝² : NonUnitalNonAssocSemiring S
    inst✝¹ : DistribMulAction R S
    inst✝ : SMulCommClass R S S
    M : Matrix m n S
    b : R
    v : n → S
    ⊢ Eq (M.mulVec (HSMul.hSMul b v)) (HSMul.hSMul b (M.mulVec v))
  -/
  ext i
  /-
    case h
    m : Type u_2
    n : Type u_3
    R : Type u_7
    S : Type u_8
    inst✝⁴ : Fintype n
    inst✝³ : Monoid R
    inst✝² : NonUnitalNonAssocSemiring S
    inst✝¹ : DistribMulAction R S
    inst✝ : SMulCommClass R S S
    M : Matrix m n S
    b : R
    v : n → S
    i : m
    ⊢ Eq (M.mulVec (HSMul.hSMul b v) i) (HSMul.hSMul b (M.mulVec v) i)
  -/
  simp only [mulVec, dotProduct, Finset.smul_sum, Pi.smul_apply, mul_smul_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem mulVec_single [Fintype n] [DecidableEq n] [NonUnitalNonAssocSemiring R] (M : Matrix m n R)
    (j : n) (x : R) : M *ᵥ Pi.single j x = fun i => M i j * x :=
  funext fun _ => dotProduct_single _ _ _


@[simp]
theorem single_vecMul [Fintype m] [DecidableEq m] [NonUnitalNonAssocSemiring R] (M : Matrix m n R)
    (i : m) (x : R) : Pi.single i x ᵥ* M = fun j => x * M i j :=
  funext fun _ => single_dotProduct _ _ _


theorem mulVec_single_one [Fintype n] [DecidableEq n] [NonAssocSemiring R]
    (M : Matrix m n R) (j : n) :
                                    /-
                                      m : Type u_2
                                      n : Type u_3
                                      R : Type u_7
                                      inst✝² : Fintype n
                                      inst✝¹ : DecidableEq n
                                      inst✝ : NonAssocSemiring R
                                      M : Matrix m n R
                                      j : n
                                      ⊢ Eq (M.mulVec (Pi.single j 1)) (M.transpose j)
                                    -/
    M *ᵥ Pi.single j 1 = Mᵀ j := by ext; simp
                                         /-
                                           🎉 no goals
                                         -/


theorem single_one_vecMul [Fintype m] [DecidableEq m] [NonAssocSemiring R]
    (i : m) (M : Matrix m n R) :
                                   /-
                                     m : Type u_2
                                     n : Type u_3
                                     R : Type u_7
                                     inst✝² : Fintype m
                                     inst✝¹ : DecidableEq m
                                     inst✝ : NonAssocSemiring R
                                     i : m
                                     M : Matrix m n R
                                     ⊢ Eq (Matrix.vecMul (Pi.single i 1) M) (M i)
                                   -/
    Pi.single i 1 ᵥ* M = M i := by simp
                                   /-
                                     🎉 no goals
                                   -/

-- @[simp] -- Porting note: not in simpNF

theorem diagonal_mulVec_single [Fintype n] [DecidableEq n] [NonUnitalNonAssocSemiring R] (v : n → R)
    (j : n) (x : R) : diagonal v *ᵥ Pi.single j x = Pi.single j (v j * x) := by
  /-
    n : Type u_3
    R : Type u_7
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : NonUnitalNonAssocSemiring R
    v : n → R
    j : n
    x : R
    ⊢ Eq ((Matrix.diagonal v).mulVec (Pi.single j x)) (Pi.single j (HMul.hMul (v j …
  -/
  ext i
  /-
    case h
    n : Type u_3
    R : Type u_7
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : NonUnitalNonAssocSemiring R
    v : n → R
    j : n
    x : R
    i : n
    ⊢ Eq ((Matrix.diagonal v).mulVec (Pi.single j x) i) (Pi.single j (HMul.hMul (v …
  -/
  rw [mulVec_diagonal]
  /-
    case h
    n : Type u_3
    R : Type u_7
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : NonUnitalNonAssocSemiring R
    v : n → R
    j : n
    x : R
    i : n
    ⊢ Eq (HMul.hMul (v i) (Pi.single j x i)) (Pi.single j (HMul.hMul (v j) x) i)
  -/
  exact Pi.apply_single (fun i x => v i * x) (fun i => mul_zero _) j x i
  /-
    🎉 no goals
  -/

-- @[simp] -- Porting note: not in simpNF

theorem single_vecMul_diagonal [Fintype n] [DecidableEq n] [NonUnitalNonAssocSemiring R] (v : n → R)
    (j : n) (x : R) : (Pi.single j x) ᵥ* (diagonal v) = Pi.single j (x * v j) := by
  /-
    n : Type u_3
    R : Type u_7
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : NonUnitalNonAssocSemiring R
    v : n → R
    j : n
    x : R
    ⊢ Eq (Matrix.vecMul (Pi.single j x) (Matrix.diagonal v)) (Pi.single j (HMul.hM …
  -/
  ext i
  /-
    case h
    n : Type u_3
    R : Type u_7
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : NonUnitalNonAssocSemiring R
    v : n → R
    j : n
    x : R
    i : n
    ⊢ Eq (Matrix.vecMul (Pi.single j x) (Matrix.diagonal v) i) (Pi.single j (HMul. …
  -/
  rw [vecMul_diagonal]
  /-
    case h
    n : Type u_3
    R : Type u_7
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : NonUnitalNonAssocSemiring R
    v : n → R
    j : n
    x : R
    i : n
    ⊢ Eq (HMul.hMul (Pi.single j x i) (v i)) (Pi.single j (HMul.hMul x (v j)) i)
  -/
  exact Pi.apply_single (fun i x => x * v i) (fun i => zero_mul _) j x i
  /-
    🎉 no goals
  -/


@[simp]
theorem vecMul_vecMul [Fintype n] [Fintype m] (v : m → α) (M : Matrix m n α) (N : Matrix n o α) :
    v ᵥ* M ᵥ* N = v ᵥ* (M * N) := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝² : NonUnitalSemiring α
    inst✝¹ : Fintype n
    inst✝ : Fintype m
    v : m → α
    M : Matrix m n α
    N : Matrix n o α
    ⊢ Eq (Matrix.vecMul (Matrix.vecMul v M) N) (Matrix.vecMul v (HMul.hMul M N))
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝² : NonUnitalSemiring α
    inst✝¹ : Fintype n
    inst✝ : Fintype m
    v : m → α
    M : Matrix m n α
    N : Matrix n o α
    x✝ : o
    ⊢ Eq (Matrix.vecMul (Matrix.vecMul v M) N x✝) (Matrix.vecMul v (HMul.hMul M N) …
  -/
  apply dotProduct_assoc
  /-
    🎉 no goals
  -/


@[simp]
theorem mulVec_mulVec [Fintype n] [Fintype o] (v : o → α) (M : Matrix m n α) (N : Matrix n o α) :
    M *ᵥ N *ᵥ v = (M * N) *ᵥ v := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝² : NonUnitalSemiring α
    inst✝¹ : Fintype n
    inst✝ : Fintype o
    v : o → α
    M : Matrix m n α
    N : Matrix n o α
    ⊢ Eq (M.mulVec (N.mulVec v)) ((HMul.hMul M N).mulVec v)
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝² : NonUnitalSemiring α
    inst✝¹ : Fintype n
    inst✝ : Fintype o
    v : o → α
    M : Matrix m n α
    N : Matrix n o α
    x✝ : m
    ⊢ Eq (M.mulVec (N.mulVec v) x✝) ((HMul.hMul M N).mulVec v x✝)
  -/
  symm
  /-
    case h
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝² : NonUnitalSemiring α
    inst✝¹ : Fintype n
    inst✝ : Fintype o
    v : o → α
    M : Matrix m n α
    N : Matrix n o α
    x✝ : m
    ⊢ Eq ((HMul.hMul M N).mulVec v x✝) (M.mulVec (N.mulVec v) x✝)
  -/
  apply dotProduct_assoc
  /-
    🎉 no goals
  -/


theorem mul_mul_apply [Fintype n] (A B C : Matrix n n α) (i j : n) :
    (A * B * C) i j = A i ⬝ᵥ B *ᵥ (Cᵀ j) := by
  /-
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalSemiring α
    inst✝ : Fintype n
    A B C : Matrix n n α
    i j : n
    ⊢ Eq (HMul.hMul (HMul.hMul A B) C i j) (dotProduct (A i) (B.mulVec (C.transpos …
  -/
  rw [Matrix.mul_assoc]
  /-
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalSemiring α
    inst✝ : Fintype n
    A B C : Matrix n n α
    i j : n
    ⊢ Eq (HMul.hMul A (HMul.hMul B C) i j) (dotProduct (A i) (B.mulVec (C.transpos …
  -/
  simp only [mul_apply, dotProduct, mulVec]
  /-
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalSemiring α
    inst✝ : Fintype n
    A B C : Matrix n n α
    i j : n
    ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (A i x) (Finset.univ.sum fun j_1 => H …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mulVec_one [Fintype n] (A : Matrix m n α) : A *ᵥ 1 = fun i => ∑ j, A i j := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonAssocSemiring α
    inst✝ : Fintype n
    A : Matrix m n α
    ⊢ Eq (A.mulVec 1) fun i => Finset.univ.sum fun j => A i j
  -/
  ext; simp [mulVec, dotProduct]
       /-
         🎉 no goals
       -/


theorem vec_one_mul [Fintype m] (A : Matrix m n α) : 1 ᵥ* A = fun j => ∑ i, A i j := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonAssocSemiring α
    inst✝ : Fintype m
    A : Matrix m n α
    ⊢ Eq (Matrix.vecMul 1 A) fun j => Finset.univ.sum fun i => A i j
  -/
  ext; simp [vecMul, dotProduct]
       /-
         🎉 no goals
       -/


@[simp]
theorem one_mulVec (v : m → α) : 1 *ᵥ v = v := by
  /-
    m : Type u_2
    α : Type v
    inst✝² : NonAssocSemiring α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    v : m → α
    ⊢ Eq (Matrix.mulVec 1 v) v
  -/
  ext
  /-
    case h
    m : Type u_2
    α : Type v
    inst✝² : NonAssocSemiring α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    v : m → α
    x✝ : m
    ⊢ Eq (Matrix.mulVec 1 v x✝) (v x✝)
  -/
  rw [← diagonal_one, mulVec_diagonal, one_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem vecMul_one (v : m → α) : v ᵥ* 1 = v := by
  /-
    m : Type u_2
    α : Type v
    inst✝² : NonAssocSemiring α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    v : m → α
    ⊢ Eq (Matrix.vecMul v 1) v
  -/
  ext
  /-
    case h
    m : Type u_2
    α : Type v
    inst✝² : NonAssocSemiring α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    v : m → α
    x✝ : m
    ⊢ Eq (Matrix.vecMul v 1 x✝) (v x✝)
  -/
  rw [← diagonal_one, vecMul_diagonal, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem diagonal_const_mulVec (x : α) (v : m → α) :
    (diagonal fun _ => x) *ᵥ v = x • v := by
  /-
    m : Type u_2
    α : Type v
    inst✝² : NonAssocSemiring α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    x : α
    v : m → α
    ⊢ Eq ((Matrix.diagonal fun x_1 => x).mulVec v) (HSMul.hSMul x v)
  -/
  ext; simp [mulVec_diagonal]
       /-
         🎉 no goals
       -/


@[simp]
theorem vecMul_diagonal_const (x : α) (v : m → α) :
    v ᵥ* (diagonal fun _ => x) = MulOpposite.op x • v := by
  /-
    m : Type u_2
    α : Type v
    inst✝² : NonAssocSemiring α
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    x : α
    v : m → α
    ⊢ Eq (Matrix.vecMul v (Matrix.diagonal fun x_1 => x)) (HSMul.hSMul (MulOpposit …
  -/
  ext; simp [vecMul_diagonal]
       /-
         🎉 no goals
       -/


@[simp]
theorem natCast_mulVec (x : ℕ) (v : m → α) : x *ᵥ v = (x : α) • v :=
  diagonal_const_mulVec _ _


@[simp]
theorem vecMul_natCast (x : ℕ) (v : m → α) : v ᵥ* x = MulOpposite.op (x : α) • v :=
  vecMul_diagonal_const _ _


-- See note [no_index around OfNat.ofNat]

@[simp]
theorem ofNat_mulVec (x : ℕ) [x.AtLeastTwo] (v : m → α) :
    OfNat.ofNat (no_index x) *ᵥ v = (OfNat.ofNat x : α) • v :=
  natCast_mulVec _ _

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem vecMul_ofNat (x : ℕ) [x.AtLeastTwo] (v : m → α) :
    v ᵥ* OfNat.ofNat (no_index x) = MulOpposite.op (OfNat.ofNat x : α) • v :=
  vecMul_natCast _ _


theorem neg_vecMul [Fintype m] (v : m → α) (A : Matrix m n α) : (-v) ᵥ* A = - (v ᵥ* A) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype m
    v : m → α
    A : Matrix m n α
    ⊢ Eq (Matrix.vecMul (Neg.neg v) A) (Neg.neg (Matrix.vecMul v A))
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype m
    v : m → α
    A : Matrix m n α
    x✝ : n
    ⊢ Eq (Matrix.vecMul (Neg.neg v) A x✝) (Neg.neg (Matrix.vecMul v A) x✝)
  -/
  apply neg_dotProduct
  /-
    🎉 no goals
  -/


theorem vecMul_neg [Fintype m] (v : m → α) (A : Matrix m n α) : v ᵥ* (-A) = - (v ᵥ* A) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype m
    v : m → α
    A : Matrix m n α
    ⊢ Eq (Matrix.vecMul v (Neg.neg A)) (Neg.neg (Matrix.vecMul v A))
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype m
    v : m → α
    A : Matrix m n α
    x✝ : n
    ⊢ Eq (Matrix.vecMul v (Neg.neg A) x✝) (Neg.neg (Matrix.vecMul v A) x✝)
  -/
  apply dotProduct_neg
  /-
    🎉 no goals
  -/


lemma neg_vecMul_neg [Fintype m] (v : m → α) (A : Matrix m n α) : (-v) ᵥ* (-A) = v ᵥ* A := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype m
    v : m → α
    A : Matrix m n α
    ⊢ Eq (Matrix.vecMul (Neg.neg v) (Neg.neg A)) (Matrix.vecMul v A)
  -/
  rw [vecMul_neg, neg_vecMul, neg_neg]
  /-
    🎉 no goals
  -/


theorem neg_mulVec [Fintype n] (v : n → α) (A : Matrix m n α) : (-A) *ᵥ v = - (A *ᵥ v) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype n
    v : n → α
    A : Matrix m n α
    ⊢ Eq ((Neg.neg A).mulVec v) (Neg.neg (A.mulVec v))
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype n
    v : n → α
    A : Matrix m n α
    x✝ : m
    ⊢ Eq ((Neg.neg A).mulVec v x✝) (Neg.neg (A.mulVec v) x✝)
  -/
  apply neg_dotProduct
  /-
    🎉 no goals
  -/


theorem mulVec_neg [Fintype n] (v : n → α) (A : Matrix m n α) : A *ᵥ (-v) = - (A *ᵥ v) := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype n
    v : n → α
    A : Matrix m n α
    ⊢ Eq (A.mulVec (Neg.neg v)) (Neg.neg (A.mulVec v))
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype n
    v : n → α
    A : Matrix m n α
    x✝ : m
    ⊢ Eq (A.mulVec (Neg.neg v) x✝) (Neg.neg (A.mulVec v) x✝)
  -/
  apply dotProduct_neg
  /-
    🎉 no goals
  -/


lemma neg_mulVec_neg [Fintype n] (v : n → α) (A : Matrix m n α) : (-A) *ᵥ (-v) = A *ᵥ v := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype n
    v : n → α
    A : Matrix m n α
    ⊢ Eq ((Neg.neg A).mulVec (Neg.neg v)) (A.mulVec v)
  -/
  rw [mulVec_neg, neg_mulVec, neg_neg]
  /-
    🎉 no goals
  -/


theorem mulVec_sub [Fintype n] (A : Matrix m n α) (x y : n → α) :
    A *ᵥ (x - y) = A *ᵥ x - A *ᵥ y := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype n
    A : Matrix m n α
    x y : n → α
    ⊢ Eq (A.mulVec (HSub.hSub x y)) (HSub.hSub (A.mulVec x) (A.mulVec y))
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype n
    A : Matrix m n α
    x y : n → α
    x✝ : m
    ⊢ Eq (A.mulVec (HSub.hSub x y) x✝) (HSub.hSub (A.mulVec x) (A.mulVec y) x✝)
  -/
  apply dotProduct_sub
  /-
    🎉 no goals
  -/


theorem sub_mulVec [Fintype n] (A B : Matrix m n α) (x : n → α) :
                                         /-
                                           m : Type u_2
                                           n : Type u_3
                                           α : Type v
                                           inst✝¹ : NonUnitalNonAssocRing α
                                           inst✝ : Fintype n
                                           A B : Matrix m n α
                                           x : n → α
                                           ⊢ Eq ((HSub.hSub A B).mulVec x) (HSub.hSub (A.mulVec x) (B.mulVec x))
                                         -/
    (A - B) *ᵥ x = A *ᵥ x - B *ᵥ x := by simp [sub_eq_add_neg, add_mulVec, neg_mulVec]
                                         /-
                                           🎉 no goals
                                         -/


theorem vecMul_sub [Fintype m] (A B : Matrix m n α) (x : m → α) :
                                         /-
                                           m : Type u_2
                                           n : Type u_3
                                           α : Type v
                                           inst✝¹ : NonUnitalNonAssocRing α
                                           inst✝ : Fintype m
                                           A B : Matrix m n α
                                           x : m → α
                                           ⊢ Eq (Matrix.vecMul x (HSub.hSub A B)) (HSub.hSub (Matrix.vecMul x A) (Matrix. …
                                         -/
    x ᵥ* (A - B) = x ᵥ* A - x ᵥ* B := by simp [sub_eq_add_neg, vecMul_add, vecMul_neg]
                                         /-
                                           🎉 no goals
                                         -/


theorem sub_vecMul [Fintype m] (A : Matrix m n α) (x y : m → α) :
    (x - y) ᵥ* A = x ᵥ* A - y ᵥ* A := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype m
    A : Matrix m n α
    x y : m → α
    ⊢ Eq (Matrix.vecMul (HSub.hSub x y) A) (HSub.hSub (Matrix.vecMul x A) (Matrix. …
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalNonAssocRing α
    inst✝ : Fintype m
    A : Matrix m n α
    x y : m → α
    x✝ : n
    ⊢ Eq (Matrix.vecMul (HSub.hSub x y) A x✝) (HSub.hSub (Matrix.vecMul x A) (Matr …
  -/
  apply sub_dotProduct
  /-
    🎉 no goals
  -/


theorem mulVec_transpose [Fintype m] (A : Matrix m n α) (x : m → α) : Aᵀ *ᵥ x = x ᵥ* A := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalCommSemiring α
    inst✝ : Fintype m
    A : Matrix m n α
    x : m → α
    ⊢ Eq (A.transpose.mulVec x) (Matrix.vecMul x A)
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalCommSemiring α
    inst✝ : Fintype m
    A : Matrix m n α
    x : m → α
    x✝ : n
    ⊢ Eq (A.transpose.mulVec x x✝) (Matrix.vecMul x A x✝)
  -/
  apply dotProduct_comm
  /-
    🎉 no goals
  -/


theorem vecMul_transpose [Fintype n] (A : Matrix m n α) (x : n → α) : x ᵥ* Aᵀ = A *ᵥ x := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalCommSemiring α
    inst✝ : Fintype n
    A : Matrix m n α
    x : n → α
    ⊢ Eq (Matrix.vecMul x A.transpose) (A.mulVec x)
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : NonUnitalCommSemiring α
    inst✝ : Fintype n
    A : Matrix m n α
    x : n → α
    x✝ : m
    ⊢ Eq (Matrix.vecMul x A.transpose x✝) (A.mulVec x x✝)
  -/
  apply dotProduct_comm
  /-
    🎉 no goals
  -/


theorem mulVec_vecMul [Fintype n] [Fintype o] (A : Matrix m n α) (B : Matrix o n α) (x : o → α) :
                                        /-
                                          m : Type u_2
                                          n : Type u_3
                                          o : Type u_4
                                          α : Type v
                                          inst✝² : NonUnitalCommSemiring α
                                          inst✝¹ : Fintype n
                                          inst✝ : Fintype o
                                          A : Matrix m n α
                                          B : Matrix o n α
                                          x : o → α
                                          ⊢ Eq (A.mulVec (Matrix.vecMul x B)) ((HMul.hMul A B.transpose).mulVec x)
                                        -/
    A *ᵥ (x ᵥ* B) = (A * Bᵀ) *ᵥ x := by rw [← mulVec_mulVec, mulVec_transpose]
                                        /-
                                          🎉 no goals
                                        -/


theorem vecMul_mulVec [Fintype m] [Fintype n] (A : Matrix m n α) (B : Matrix m o α) (x : n → α) :
                                        /-
                                          m : Type u_2
                                          n : Type u_3
                                          o : Type u_4
                                          α : Type v
                                          inst✝² : NonUnitalCommSemiring α
                                          inst✝¹ : Fintype m
                                          inst✝ : Fintype n
                                          A : Matrix m n α
                                          B : Matrix m o α
                                          x : n → α
                                          ⊢ Eq (Matrix.vecMul (A.mulVec x) B) (Matrix.vecMul x (HMul.hMul A.transpose B))
                                        -/
    (A *ᵥ x) ᵥ* B = x ᵥ* (Aᵀ * B) := by rw [← vecMul_vecMul, vecMul_transpose]
                                        /-
                                          🎉 no goals
                                        -/


theorem mulVec_smul_assoc [Fintype n] (A : Matrix m n α) (b : n → α) (a : α) :
    A *ᵥ (a • b) = a • A *ᵥ b := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : CommSemiring α
    inst✝ : Fintype n
    A : Matrix m n α
    b : n → α
    a : α
    ⊢ Eq (A.mulVec (HSMul.hSMul a b)) (HSMul.hSMul a (A.mulVec b))
  -/
  ext
  /-
    case h
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝¹ : CommSemiring α
    inst✝ : Fintype n
    A : Matrix m n α
    b : n → α
    a : α
    x✝ : m
    ⊢ Eq (A.mulVec (HSMul.hSMul a b) x✝) (HSMul.hSMul a (A.mulVec b) x✝)
  -/
  apply dotProduct_smul
  /-
    🎉 no goals
  -/


@[simp]
theorem intCast_mulVec (x : ℤ) (v : m → α) : x *ᵥ v = (x : α) • v :=
  diagonal_const_mulVec _ _


@[simp]
theorem vecMul_intCast (x : ℤ) (v : m → α) : v ᵥ* x = MulOpposite.op (x : α) • v :=
  vecMul_diagonal_const _ _


@[simp]
theorem transpose_mul [AddCommMonoid α] [CommSemigroup α] [Fintype n] (M : Matrix m n α)
    (N : Matrix n l α) : (M * N)ᵀ = Nᵀ * Mᵀ := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : AddCommMonoid α
    inst✝¹ : CommSemigroup α
    inst✝ : Fintype n
    M : Matrix m n α
    N : Matrix n l α
    ⊢ Eq (HMul.hMul M N).transpose (HMul.hMul N.transpose M.transpose)
  -/
  ext
  /-
    case a
    l : Type u_1
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝² : AddCommMonoid α
    inst✝¹ : CommSemigroup α
    inst✝ : Fintype n
    M : Matrix m n α
    N : Matrix n l α
    i✝ : l
    j✝ : m
    ⊢ Eq ((HMul.hMul M N).transpose i✝ j✝) (HMul.hMul N.transpose M.transpose i✝ j✝)
  -/
  apply dotProduct_comm
  /-
    🎉 no goals
  -/


theorem submatrix_mul [Fintype n] [Fintype o] [Mul α] [AddCommMonoid α] {p q : Type*}
    (M : Matrix m n α) (N : Matrix n p α) (e₁ : l → m) (e₂ : o → n) (e₃ : q → p)
    (he₂ : Function.Bijective e₂) :
    (M * N).submatrix e₁ e₃ = M.submatrix e₁ e₂ * N.submatrix e₂ e₃ :=
  ext fun _ _ => (he₂.sum_comp _).symm


@[simp]
theorem submatrix_mul_equiv [Fintype n] [Fintype o] [AddCommMonoid α] [Mul α] {p q : Type*}
    (M : Matrix m n α) (N : Matrix n p α) (e₁ : l → m) (e₂ : o ≃ n) (e₃ : q → p) :
    M.submatrix e₁ e₂ * N.submatrix e₂ e₃ = (M * N).submatrix e₁ e₃ :=
  (submatrix_mul M N e₁ e₂ e₃ e₂.bijective).symm


theorem submatrix_mulVec_equiv [Fintype n] [Fintype o] [NonUnitalNonAssocSemiring α]
    (M : Matrix m n α) (v : o → α) (e₁ : l → m) (e₂ : o ≃ n) :
    M.submatrix e₁ e₂ *ᵥ v = (M *ᵥ (v ∘ e₂.symm)) ∘ e₁ :=
  funext fun _ => Eq.symm (dotProduct_comp_equiv_symm _ _ _)


@[simp]
theorem submatrix_id_mul_left [Fintype n] [Fintype o] [Mul α] [AddCommMonoid α] {p : Type*}
    (M : Matrix m n α) (N : Matrix o p α) (e₁ : l → m) (e₂ : n ≃ o) :
    M.submatrix e₁ id * N.submatrix e₂ id = M.submatrix e₁ e₂.symm * N := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝³ : Fintype n
    inst✝² : Fintype o
    inst✝¹ : Mul α
    inst✝ : AddCommMonoid α
    p : Type u_10
    M : Matrix m n α
    N : Matrix o p α
    e₁ : l → m
    e₂ : Equiv n o
    ⊢ Eq (HMul.hMul (M.submatrix e₁ id) (N.submatrix (⇑e₂) id)) (HMul.hMul (M.subm …
  -/
  ext; simp [mul_apply, ← e₂.bijective.sum_comp]
       /-
         🎉 no goals
       -/


@[simp]
theorem submatrix_id_mul_right [Fintype n] [Fintype o] [Mul α] [AddCommMonoid α] {p : Type*}
    (M : Matrix m n α) (N : Matrix o p α) (e₁ : l → p) (e₂ : o ≃ n) :
    M.submatrix id e₂ * N.submatrix id e₁ = M * N.submatrix e₂.symm e₁ := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝³ : Fintype n
    inst✝² : Fintype o
    inst✝¹ : Mul α
    inst✝ : AddCommMonoid α
    p : Type u_10
    M : Matrix m n α
    N : Matrix o p α
    e₁ : l → p
    e₂ : Equiv o n
    ⊢ Eq (HMul.hMul (M.submatrix id ⇑e₂) (N.submatrix id e₁)) (HMul.hMul M (N.subm …
  -/
  ext; simp [mul_apply, ← e₂.bijective.sum_comp]
       /-
         🎉 no goals
       -/


theorem submatrix_vecMul_equiv [Fintype l] [Fintype m] [NonUnitalNonAssocSemiring α]
    (M : Matrix m n α) (v : l → α) (e₁ : l ≃ m) (e₂ : o → n) :
    v ᵥ* M.submatrix e₁ e₂ = ((v ∘ e₁.symm) ᵥ* M) ∘ e₂ :=
  funext fun _ => Eq.symm (comp_equiv_symm_dotProduct _ _ _)


theorem mul_submatrix_one [Fintype n] [Finite o] [NonAssocSemiring α] [DecidableEq o] (e₁ : n ≃ o)
    (e₂ : l → o) (M : Matrix m n α) :
    M * (1 : Matrix o o α).submatrix e₁ e₂ = submatrix M id (e₁.symm ∘ e₂) := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝³ : Fintype n
    inst✝² : Finite o
    inst✝¹ : NonAssocSemiring α
    inst✝ : DecidableEq o
    e₁ : Equiv n o
    e₂ : l → o
    M : Matrix m n α
    ⊢ Eq (HMul.hMul M (Matrix.submatrix 1 (⇑e₁) e₂)) (M.submatrix id (Function.com …
  -/
  cases nonempty_fintype o
  /-
    case intro
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝³ : Fintype n
    inst✝² : Finite o
    inst✝¹ : NonAssocSemiring α
    inst✝ : DecidableEq o
    e₁ : Equiv n o
    e₂ : l → o
    M : Matrix m n α
    val✝ : Fintype o
    ⊢ Eq (HMul.hMul M (Matrix.submatrix 1 (⇑e₁) e₂)) (M.submatrix id (Function.com …
  -/
  let A := M.submatrix id e₁.symm
  have : M = A.submatrix id e₁ := by
    simp only [A, submatrix_submatrix, Function.comp_id, submatrix_id_id, Equiv.symm_comp_self]
  /-
    case intro
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝³ : Fintype n
    inst✝² : Finite o
    inst✝¹ : NonAssocSemiring α
    inst✝ : DecidableEq o
    e₁ : Equiv n o
    e₂ : l → o
    M : Matrix m n α
    val✝ : Fintype o
    A : Matrix m o α := M.submatrix id ⇑e₁.symm
    this : Eq M (A.submatrix id ⇑e₁)
    ⊢ Eq (HMul.hMul M (Matrix.submatrix 1 (⇑e₁) e₂)) (M.submatrix id (Function.com …
  -/
  rw [this, submatrix_mul_equiv]
  simp only [A, Matrix.mul_one, submatrix_submatrix, Function.comp_id, submatrix_id_id,
    Equiv.symm_comp_self]


theorem one_submatrix_mul [Fintype m] [Finite o] [NonAssocSemiring α] [DecidableEq o] (e₁ : l → o)
    (e₂ : m ≃ o) (M : Matrix m n α) :
    ((1 : Matrix o o α).submatrix e₁ e₂) * M = submatrix M (e₂.symm ∘ e₁) id := by
  /-
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝³ : Fintype m
    inst✝² : Finite o
    inst✝¹ : NonAssocSemiring α
    inst✝ : DecidableEq o
    e₁ : l → o
    e₂ : Equiv m o
    M : Matrix m n α
    ⊢ Eq (HMul.hMul (Matrix.submatrix 1 e₁ ⇑e₂) M) (M.submatrix (Function.comp (⇑e …
  -/
  cases nonempty_fintype o
  /-
    case intro
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝³ : Fintype m
    inst✝² : Finite o
    inst✝¹ : NonAssocSemiring α
    inst✝ : DecidableEq o
    e₁ : l → o
    e₂ : Equiv m o
    M : Matrix m n α
    val✝ : Fintype o
    ⊢ Eq (HMul.hMul (Matrix.submatrix 1 e₁ ⇑e₂) M) (M.submatrix (Function.comp (⇑e …
  -/
  let A := M.submatrix e₂.symm id
  have : M = A.submatrix e₂ id := by
    simp only [A, submatrix_submatrix, Function.comp_id, submatrix_id_id, Equiv.symm_comp_self]
  /-
    case intro
    l : Type u_1
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    inst✝³ : Fintype m
    inst✝² : Finite o
    inst✝¹ : NonAssocSemiring α
    inst✝ : DecidableEq o
    e₁ : l → o
    e₂ : Equiv m o
    M : Matrix m n α
    val✝ : Fintype o
    A : Matrix o n α := M.submatrix (⇑e₂.symm) id
    this : Eq M (A.submatrix (⇑e₂) id)
    ⊢ Eq (HMul.hMul (Matrix.submatrix 1 e₁ ⇑e₂) M) (M.submatrix (Function.comp (⇑e …
  -/
  rw [this, submatrix_mul_equiv]
  simp only [A, Matrix.one_mul, submatrix_submatrix, Function.comp_id, submatrix_id_id,
    Equiv.symm_comp_self]


theorem submatrix_mul_transpose_submatrix [Fintype m] [Fintype n] [AddCommMonoid α] [Mul α]
    (e : m ≃ n) (M : Matrix m n α) : M.submatrix id e * Mᵀ.submatrix e id = M * Mᵀ := by
  /-
    m : Type u_2
    n : Type u_3
    α : Type v
    inst✝³ : Fintype m
    inst✝² : Fintype n
    inst✝¹ : AddCommMonoid α
    inst✝ : Mul α
    e : Equiv m n
    M : Matrix m n α
    ⊢ Eq (HMul.hMul (M.submatrix id ⇑e) (M.transpose.submatrix (⇑e) id)) (HMul.hMu …
  -/
  rw [submatrix_mul_equiv, submatrix_id_id]
  /-
    🎉 no goals
  -/


theorem map_matrix_mul (M : Matrix m n α) (N : Matrix n o α) (i : m) (j : o) (f : α →+* β) :
    f ((M * N) i j) = (M.map f * N.map f) i j := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    α : Type v
    β : Type w
    inst✝² : Fintype n
    inst✝¹ : NonAssocSemiring α
    inst✝ : NonAssocSemiring β
    M : Matrix m n α
    N : Matrix n o α
    i : m
    j : o
    f : RingHom α β
    ⊢ Eq (f (HMul.hMul M N i j)) (HMul.hMul (M.map ⇑f) (N.map ⇑f) i j)
  -/
  simp [Matrix.mul_apply, map_sum]
  /-
    🎉 no goals
  -/


theorem map_dotProduct [NonAssocSemiring R] [NonAssocSemiring S] (f : R →+* S) (v w : n → R) :
    f (v ⬝ᵥ w) = f ∘ v ⬝ᵥ f ∘ w := by
  /-
    n : Type u_3
    R : Type u_7
    S : Type u_8
    inst✝² : Fintype n
    inst✝¹ : NonAssocSemiring R
    inst✝ : NonAssocSemiring S
    f : RingHom R S
    v w : n → R
    ⊢ Eq (f (dotProduct v w)) (dotProduct (Function.comp (⇑f) v) (Function.comp (⇑ …
  -/
  simp only [dotProduct, map_sum f, f.map_mul, Function.comp]
  /-
    🎉 no goals
  -/


theorem map_vecMul [NonAssocSemiring R] [NonAssocSemiring S] (f : R →+* S) (M : Matrix n m R)
    (v : n → R) (i : m) : f ((v ᵥ* M) i) =  ((f ∘ v) ᵥ* M.map f) i := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_7
    S : Type u_8
    inst✝² : Fintype n
    inst✝¹ : NonAssocSemiring R
    inst✝ : NonAssocSemiring S
    f : RingHom R S
    M : Matrix n m R
    v : n → R
    i : m
    ⊢ Eq (f (Matrix.vecMul v M i)) (Matrix.vecMul (Function.comp (⇑f) v) (M.map ⇑f …
  -/
  simp only [Matrix.vecMul, Matrix.map_apply, RingHom.map_dotProduct, Function.comp_def]
  /-
    🎉 no goals
  -/


theorem map_mulVec [NonAssocSemiring R] [NonAssocSemiring S] (f : R →+* S) (M : Matrix m n R)
    (v : n → R) (i : m) : f ((M *ᵥ v) i) = (M.map f *ᵥ (f ∘ v)) i := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_7
    S : Type u_8
    inst✝² : Fintype n
    inst✝¹ : NonAssocSemiring R
    inst✝ : NonAssocSemiring S
    f : RingHom R S
    M : Matrix m n R
    v : n → R
    i : m
    ⊢ Eq (f (M.mulVec v i)) ((M.map ⇑f).mulVec (Function.comp (⇑f) v) i)
  -/
  simp only [Matrix.mulVec, Matrix.map_apply, RingHom.map_dotProduct, Function.comp_def]
  /-
    🎉 no goals
  -/


