/-- The canonical embedding from a tensor power to the tensor algebra -/
def toTensorAlgebra {n} : ⨂[R]^n M →ₗ[R] TensorAlgebra R M :=
  PiTensorProduct.lift (TensorAlgebra.tprod R M n)


@[simp]
theorem toTensorAlgebra_tprod {n} (x : Fin n → M) :
    TensorPower.toTensorAlgebra (PiTensorProduct.tprod R x) = TensorAlgebra.tprod R M n x :=
  PiTensorProduct.lift.tprod _


@[simp]
theorem toTensorAlgebra_gOne :
    TensorPower.toTensorAlgebra (@GradedMonoid.GOne.one _ (fun n => ⨂[R]^n M) _ _) = 1 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Eq (TensorPower.toTensorAlgebra GradedMonoid.GOne.one) 1
  -/
  simp [GradedMonoid.GOne.one, TensorPower.toTensorAlgebra_tprod]
  /-
    🎉 no goals
  -/


@[simp]
theorem toTensorAlgebra_gMul {i j} (a : (⨂[R]^i) M) (b : (⨂[R]^j) M) :
    TensorPower.toTensorAlgebra (@GradedMonoid.GMul.mul _ (fun n => ⨂[R]^n M) _ _ _ _ a b) =
      TensorPower.toTensorAlgebra a * TensorPower.toTensorAlgebra b := by
  -- change `a` and `b` to `tprod R a` and `tprod R b`
  rw [TensorPower.gMul_eq_coe_linearMap, ← LinearMap.compr₂_apply, ← @LinearMap.mul_apply' R, ←
    LinearMap.compl₂_apply, ← LinearMap.comp_apply]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    i j : Nat
    a : TensorPower R i M
    b : TensorPower R j M
    ⊢ Eq (((((TensorProduct.mk R (TensorPower R i M) (TensorPower R j M)).compr₂ ↑ …
  -/
  refine LinearMap.congr_fun (LinearMap.congr_fun ?_ a) b
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    i j : Nat
    a : TensorPower R i M
    b : TensorPower R j M
    ⊢ Eq (((TensorProduct.mk R (TensorPower R i M) (TensorPower R j M)).compr₂ ↑Te …
  -/
  clear! a b
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    i j : Nat
    ⊢ Eq (((TensorProduct.mk R (TensorPower R i M) (TensorPower R j M)).compr₂ ↑Te …
  -/
  ext (a b)
  -- Porting note: pulled the next two lines out of the long `simp only` below.
  /-
    case H.H.H.H
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    i j : Nat
    a : Fin i → M
    b : Fin j → M
    ⊢ Eq (((((((TensorProduct.mk R (TensorPower R i M) (TensorPower R j M)).compr₂ …
  -/
  simp only [LinearMap.compMultilinearMap_apply]
  /-
    case H.H.H.H
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    i j : Nat
    a : Fin i → M
    b : Fin j → M
    ⊢ Eq (((((TensorProduct.mk R (TensorPower R i M) (TensorPower R j M)).compr₂ ↑ …
  -/
  rw [LinearMap.compr₂_apply, ← gMul_eq_coe_linearMap]
  simp only [LinearMap.compr₂_apply, LinearMap.mul_apply', LinearMap.compl₂_apply,
    LinearMap.comp_apply, LinearMap.compMultilinearMap_apply, PiTensorProduct.lift.tprod,
    TensorPower.tprod_mul_tprod, TensorPower.toTensorAlgebra_tprod, TensorAlgebra.tprod_apply, ←
    gMul_eq_coe_linearMap]
  /-
    case H.H.H.H
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    i j : Nat
    a : Fin i → M
    b : Fin j → M
    ⊢ Eq (List.ofFn fun i_1 => (TensorAlgebra.ι R) (Fin.append a b i_1)).prod (HMu …
  -/
  refine Eq.trans ?_ List.prod_append
  /-
    case H.H.H.H
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    i j : Nat
    a : Fin i → M
    b : Fin j → M
    ⊢ Eq (List.ofFn fun i_1 => (TensorAlgebra.ι R) (Fin.append a b i_1)).prod (HAp …
  -/
  congr
  -- Porting note: `erw` for `Function.comp`
  erw [← List.map_ofFn _ (TensorAlgebra.ι R), ← List.map_ofFn _ (TensorAlgebra.ι R), ←
    List.map_ofFn _ (TensorAlgebra.ι R), ← List.map_append, List.ofFn_fin_append]


@[simp]
theorem toTensorAlgebra_galgebra_toFun (r : R) :
    TensorPower.toTensorAlgebra (DirectSum.GAlgebra.toFun (R := R) (A := fun n => ⨂[R]^n M) r) =
      algebraMap _ _ r := by
  rw [TensorPower.galgebra_toFun_def, TensorPower.algebraMap₀_eq_smul_one, LinearMap.map_smul,
    TensorPower.toTensorAlgebra_gOne, Algebra.algebraMap_eq_smul_one]


/-- The canonical map from a direct sum of tensor powers to the tensor algebra. -/
def ofDirectSum : (⨁ n, ⨂[R]^n M) →ₐ[R] TensorAlgebra R M :=
  DirectSum.toAlgebra _ _ (fun _ => TensorPower.toTensorAlgebra) TensorPower.toTensorAlgebra_gOne
    (fun {_ _} => TensorPower.toTensorAlgebra_gMul)


@[simp]
theorem ofDirectSum_of_tprod {n} (x : Fin n → M) :
    ofDirectSum (DirectSum.of _ n (PiTensorProduct.tprod R x)) = tprod R M n x :=
  (DirectSum.toAddMonoid_of
    (fun _ ↦ LinearMap.toAddMonoidHom TensorPower.toTensorAlgebra) _ _).trans
  (TensorPower.toTensorAlgebra_tprod _)


/-- The canonical map from the tensor algebra to a direct sum of tensor powers. -/
def toDirectSum : TensorAlgebra R M →ₐ[R] ⨁ n, ⨂[R]^n M :=
  TensorAlgebra.lift R <|
    DirectSum.lof R ℕ (fun n => ⨂[R]^n M) _ ∘ₗ
      (LinearEquiv.symm <| PiTensorProduct.subsingletonEquiv (0 : Fin 1) : M ≃ₗ[R] _).toLinearMap


@[simp]
theorem toDirectSum_ι (x : M) :
    toDirectSum (ι R x) =
      DirectSum.of (fun n => ⨂[R]^n M) _ (PiTensorProduct.tprod R fun _ : Fin 1 => x) :=
  TensorAlgebra.lift_ι_apply _ _


theorem ofDirectSum_comp_toDirectSum :
    ofDirectSum.comp toDirectSum = AlgHom.id R (TensorAlgebra R M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Eq (TensorAlgebra.ofDirectSum.comp TensorAlgebra.toDirectSum) (AlgHom.id R ( …
  -/
  ext
  /-
    case w.h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x✝ : M
    ⊢ Eq (((TensorAlgebra.ofDirectSum.comp TensorAlgebra.toDirectSum).toLinearMap. …
  -/
  simp [DirectSum.lof_eq_of, tprod_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofDirectSum_toDirectSum (x : TensorAlgebra R M) :
    ofDirectSum (TensorAlgebra.toDirectSum x) = x :=
  AlgHom.congr_fun ofDirectSum_comp_toDirectSum x

-- See https://github.com/leanprover-community/batteries/issues/365 for the simpNF issue.

@[simp, nolint simpNF]
theorem mk_reindex_cast {n m : ℕ} (h : n = m) (x : ⨂[R]^n M) :
    GradedMonoid.mk (A := fun i => (⨂[R]^i) M) m
    (PiTensorProduct.reindex R (fun _ ↦ M) (Equiv.cast <| congr_arg Fin h) x) =
    GradedMonoid.mk n x :=
  Eq.symm (PiTensorProduct.gradedMonoid_eq_of_reindex_cast h rfl)


@[simp]
theorem mk_reindex_fin_cast {n m : ℕ} (h : n = m) (x : ⨂[R]^n M) :
    GradedMonoid.mk (A := fun i => (⨂[R]^i) M) m
    (PiTensorProduct.reindex R (fun _ ↦ M) (finCongr h) x) = GradedMonoid.mk n x := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    n m : Nat
    h : Eq n m
    x : TensorPower R n M
    ⊢ Eq (GradedMonoid.mk m ((PiTensorProduct.reindex R (fun x => M) (finCongr h)) …
  -/
  rw [finCongr_eq_equivCast, mk_reindex_cast h]
  /-
    🎉 no goals
  -/


/-- The product of tensor products made of a single vector is the same as a single product of
all the vectors. -/
theorem _root_.TensorPower.list_prod_gradedMonoid_mk_single (n : ℕ) (x : Fin n → M) :
    ((List.finRange n).map fun a =>
          (GradedMonoid.mk _ (PiTensorProduct.tprod R fun _ : Fin 1 => x a) :
            GradedMonoid fun n => ⨂[R]^n M)).prod =
      GradedMonoid.mk n (PiTensorProduct.tprod R x) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    n : Nat
    x : Fin n → M
    ⊢ Eq (List.map (fun a => GradedMonoid.mk 1 ((PiTensorProduct.tprod R) fun x_1  …
  -/
  refine Fin.consInduction ?_ ?_ x <;> clear x
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      n : Nat
      ⊢ Eq (List.map (fun a => GradedMonoid.mk 1 ((PiTensorProduct.tprod R) fun x => …
    -/
  · rw [List.finRange_zero, List.map_nil, List.prod_nil]
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      n : Nat
      ⊢ Eq 1 (GradedMonoid.mk 0 ((PiTensorProduct.tprod R) Fin.elim0))
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      n : Nat
      ⊢ ∀ {n : Nat} (x₀ : M) (x : Fin n → M), Eq (List.map (fun a => GradedMonoid.mk …
    -/
  · intro n x₀ x ih
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      n✝ n : Nat
      x₀ : M
      x : Fin n → M
      ih : Eq (List.map (fun a => GradedMonoid.mk 1 ((PiTensorProduct.tprod R) fun x …
      ⊢ Eq (List.map (fun a => GradedMonoid.mk 1 ((PiTensorProduct.tprod R) fun x_1  …
    -/
    rw [List.finRange_succ_eq_map, List.map_cons, List.prod_cons, List.map_map]
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      n✝ n : Nat
      x₀ : M
      x : Fin n → M
      ih : Eq (List.map (fun a => GradedMonoid.mk 1 ((PiTensorProduct.tprod R) fun x …
      ⊢ Eq (HMul.hMul (GradedMonoid.mk 1 ((PiTensorProduct.tprod R) fun x_1 => Fin.c …
    -/
    simp_rw [Function.comp_def, Fin.cons_zero, Fin.cons_succ]
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      n✝ n : Nat
      x₀ : M
      x : Fin n → M
      ih : Eq (List.map (fun a => GradedMonoid.mk 1 ((PiTensorProduct.tprod R) fun x …
      ⊢ Eq (HMul.hMul (GradedMonoid.mk 1 ((PiTensorProduct.tprod R) fun x => x₀)) (L …
    -/
    rw [ih, GradedMonoid.mk_mul_mk, TensorPower.tprod_mul_tprod]
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      n✝ n : Nat
      x₀ : M
      x : Fin n → M
      ih : Eq (List.map (fun a => GradedMonoid.mk 1 ((PiTensorProduct.tprod R) fun x …
      ⊢ Eq (GradedMonoid.mk (HAdd.hAdd 1 n) ((PiTensorProduct.tprod R) (Fin.append ( …
    -/
    refine TensorPower.gradedMonoid_eq_of_cast (add_comm _ _) ?_
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      n✝ n : Nat
      x₀ : M
      x : Fin n → M
      ih : Eq (List.map (fun a => GradedMonoid.mk 1 ((PiTensorProduct.tprod R) fun x …
      ⊢ Eq ((TensorPower.cast R M ⋯) (GradedMonoid.mk (HAdd.hAdd 1 n) ((PiTensorProd …
    -/
    dsimp only [GradedMonoid.mk]
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      n✝ n : Nat
      x₀ : M
      x : Fin n → M
      ih : Eq (List.map (fun a => GradedMonoid.mk 1 ((PiTensorProduct.tprod R) fun x …
      ⊢ Eq ((TensorPower.cast R M ⋯) ((PiTensorProduct.tprod R) (Fin.append (fun x = …
    -/
    rw [TensorPower.cast_tprod]
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      n✝ n : Nat
      x₀ : M
      x : Fin n → M
      ih : Eq (List.map (fun a => GradedMonoid.mk 1 ((PiTensorProduct.tprod R) fun x …
      ⊢ Eq ((PiTensorProduct.tprod R) (Function.comp (Fin.append (fun x => x₀) x) (F …
    -/
    simp_rw [Fin.append_left_eq_cons, Function.comp_def]
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      n✝ n : Nat
      x₀ : M
      x : Fin n → M
      ih : Eq (List.map (fun a => GradedMonoid.mk 1 ((PiTensorProduct.tprod R) fun x …
      ⊢ Eq ((PiTensorProduct.tprod R) fun x_1 => Fin.cons x₀ x (Fin.cast ⋯ (Fin.cast …
    -/
    congr 1 with i
    /-
      🎉 no goals
    -/


theorem toDirectSum_tensorPower_tprod {n} (x : Fin n → M) :
    toDirectSum (tprod R M n x) = DirectSum.of _ n (PiTensorProduct.tprod R x) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    n : Nat
    x : Fin n → M
    ⊢ Eq (TensorAlgebra.toDirectSum ((TensorAlgebra.tprod R M n) x)) ((DirectSum.o …
  -/
  rw [tprod_apply, map_list_prod, List.map_ofFn]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    n : Nat
    x : Fin n → M
    ⊢ Eq (List.ofFn (Function.comp ⇑TensorAlgebra.toDirectSum fun i => (TensorAlge …
  -/
  simp_rw [Function.comp_def, toDirectSum_ι]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    n : Nat
    x : Fin n → M
    ⊢ Eq (List.ofFn fun x_1 => (DirectSum.of (fun n => TensorPower R n M) 1) ((PiT …
  -/
  rw [DirectSum.list_prod_ofFn_of_eq_dProd]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    n : Nat
    x : Fin n → M
    ⊢ Eq ((DirectSum.of (fun n => TensorPower R n M) ((List.finRange n).dProdIndex …
  -/
  apply DirectSum.of_eq_of_gradedMonoid_eq
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    n : Nat
    x : Fin n → M
    ⊢ Eq (GradedMonoid.mk ((List.finRange n).dProdIndex fun x => 1) ((List.finRang …
  -/
  rw [GradedMonoid.mk_list_dProd]
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    n : Nat
    x : Fin n → M
    ⊢ Eq (List.map (fun a => GradedMonoid.mk 1 ((PiTensorProduct.tprod R) fun x_1  …
  -/
  rw [TensorPower.list_prod_gradedMonoid_mk_single]
  /-
    🎉 no goals
  -/


theorem toDirectSum_comp_ofDirectSum :
    toDirectSum.comp ofDirectSum = AlgHom.id R (⨁ n, ⨂[R]^n M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Eq (TensorAlgebra.toDirectSum.comp TensorAlgebra.ofDirectSum) (AlgHom.id R ( …
  -/
  ext
  /-
    case h.H.H
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    i✝ : Nat
    x✝ : Fin i✝ → M
    ⊢ Eq ((((TensorAlgebra.toDirectSum.comp TensorAlgebra.ofDirectSum).toLinearMap …
  -/
  simp [DirectSum.lof_eq_of, -tprod_apply, toDirectSum_tensorPower_tprod]
  /-
    🎉 no goals
  -/


@[simp]
theorem toDirectSum_ofDirectSum (x : ⨁ n, ⨂[R]^n M) :
    TensorAlgebra.toDirectSum (ofDirectSum x) = x :=
  AlgHom.congr_fun toDirectSum_comp_ofDirectSum x


/-- The tensor algebra is isomorphic to a direct sum of tensor powers. -/
@[simps!]
def equivDirectSum : TensorAlgebra R M ≃ₐ[R] ⨁ n, ⨂[R]^n M :=
  AlgEquiv.ofAlgHom toDirectSum ofDirectSum toDirectSum_comp_ofDirectSum
    ofDirectSum_comp_toDirectSum


