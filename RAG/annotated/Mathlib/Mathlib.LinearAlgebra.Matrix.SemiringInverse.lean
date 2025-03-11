/-- The determinant, but only the terms of a given sign. -/
def detp : R := ∑ σ ∈ ofSign s, ∏ k, A k (σ k)


@[simp]
lemma detp_one_one : detp 1 (1 : Matrix n n R) = 1 := by
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    ⊢ Eq (Matrix.detp 1 1) 1
  -/
  rw [detp, sum_eq_single_of_mem 1]
    /-
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      ⊢ Eq (Finset.univ.prod fun k => 1 k (1 k)) 1
    -/
  · simp [one_apply]
    /-
      🎉 no goals
    -/
    /-
      case h
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      ⊢ Membership.mem (Equiv.Perm.ofSign 1) 1
    -/
  · simp [ofSign]
    /-
      🎉 no goals
    -/
    /-
      case h₀
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      ⊢ ∀ (b : Equiv.Perm n), Membership.mem (Equiv.Perm.ofSign 1) b → Ne b 1 → Eq ( …
    -/
  · rintro σ - hσ1
    /-
      case h₀
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      σ : Equiv.Perm n
      hσ1 : Ne σ 1
      ⊢ Eq (Finset.univ.prod fun k => 1 k (σ k)) 0
    -/
    obtain ⟨i, hi⟩ := not_forall.mp (mt Perm.ext_iff.mpr hσ1)
    /-
      case h₀.intro
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      σ : Equiv.Perm n
      hσ1 : Ne σ 1
      i : n
      hi : Not (Eq (σ i) (1 i))
      ⊢ Eq (Finset.univ.prod fun k => 1 k (σ k)) 0
    -/
    exact prod_eq_zero (mem_univ i) (one_apply_ne' hi)
    /-
      🎉 no goals
    -/


@[simp]
lemma detp_neg_one_one : detp (-1) (1 : Matrix n n R) = 0 := by
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    ⊢ Eq (Matrix.detp (-1) 1) 0
  -/
  rw [detp, sum_eq_zero]
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    ⊢ ∀ (x : Equiv.Perm n), Membership.mem (Equiv.Perm.ofSign (-1)) x → Eq (Finset …
  -/
  intro σ hσ
  have hσ1 : σ ≠ 1 := by
    contrapose! hσ
    rw [hσ, mem_ofSign, sign_one]
    decide
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    σ : Equiv.Perm n
    hσ : Membership.mem (Equiv.Perm.ofSign (-1)) σ
    hσ1 : Ne σ 1
    ⊢ Eq (Finset.univ.prod fun k => 1 k (σ k)) 0
  -/
  obtain ⟨i, hi⟩ := not_forall.mp (mt Perm.ext_iff.mpr hσ1)
  /-
    case intro
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    σ : Equiv.Perm n
    hσ : Membership.mem (Equiv.Perm.ofSign (-1)) σ
    hσ1 : Ne σ 1
    i : n
    hi : Not (Eq (σ i) (1 i))
    ⊢ Eq (Finset.univ.prod fun k => 1 k (σ k)) 0
  -/
  exact prod_eq_zero (mem_univ i) (one_apply_ne' hi)
  /-
    🎉 no goals
  -/


/-- The adjugate matrix, but only the terms of a given sign. -/
def adjp : Matrix n n R :=
  of fun i j ↦ ∑ σ ∈ (ofSign s).filter (· j = i), ∏ k ∈ {j}ᶜ, A k (σ k)


lemma adjp_apply (i j : n) :
    adjp s A i j = ∑ σ ∈ (ofSign s).filter (· j = i), ∏ k ∈ {j}ᶜ, A k (σ k) :=
  rfl


theorem detp_mul :
    detp 1 (A * B) + (detp 1 A * detp (-1) B + detp (-1) A * detp 1 B) =
      detp (-1) (A * B) + (detp 1 A * detp 1 B + detp (-1) A * detp (-1) B) := by
  have hf {s t} {σ : Perm n} (hσ : σ ∈ ofSign s) :
      ofSign (t * s) = (ofSign t).map (mulRightEmbedding σ) := by
    ext τ
    simp_rw [mem_map, mulRightEmbedding_apply, ← eq_mul_inv_iff_mul_eq, exists_eq_right,
      mem_ofSign, _root_.map_mul, _root_.map_inv, mul_inv_eq_iff_eq_mul, mem_ofSign.mp hσ]
  have h {s t} : detp s A * detp t B =
      ∑ σ ∈ ofSign s, ∑ τ ∈ ofSign (t * s), ∏ k, A k (σ k) * B (σ k) (τ k) := by
    simp_rw [detp, sum_mul_sum, prod_mul_distrib]
    refine sum_congr rfl fun σ hσ ↦ ?_
    simp_rw [hf hσ, sum_map, mulRightEmbedding_apply, Perm.mul_apply]
    exact sum_congr rfl fun τ hτ ↦ (congr_arg (_ * ·) (Equiv.prod_comp σ _).symm)
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hf : ∀ {s t : Units Int} {σ : Equiv.Perm n}, Membership.mem (Equiv.Perm.ofSign …
    h : ∀ {s t : Units Int}, Eq (HMul.hMul (Matrix.detp s A) (Matrix.detp t B)) (( …
    ⊢ Eq (HAdd.hAdd (Matrix.detp 1 (HMul.hMul A B)) (HAdd.hAdd (HMul.hMul (Matrix. …
  -/
  let ι : Perm n ↪ (n → n) := ⟨_, coe_fn_injective⟩
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hf : ∀ {s t : Units Int} {σ : Equiv.Perm n}, Membership.mem (Equiv.Perm.ofSign …
    h : ∀ {s t : Units Int}, Eq (HMul.hMul (Matrix.detp s A) (Matrix.detp t B)) (( …
    ι : Function.Embedding (Equiv.Perm n) (n → n) := { toFun := fun e => ⇑e, inj'  …
    ⊢ Eq (HAdd.hAdd (Matrix.detp 1 (HMul.hMul A B)) (HAdd.hAdd (HMul.hMul (Matrix. …
  -/
  have hι {σ x} : ι σ x = σ x := rfl
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hf : ∀ {s t : Units Int} {σ : Equiv.Perm n}, Membership.mem (Equiv.Perm.ofSign …
    h : ∀ {s t : Units Int}, Eq (HMul.hMul (Matrix.detp s A) (Matrix.detp t B)) (( …
    ι : Function.Embedding (Equiv.Perm n) (n → n) := { toFun := fun e => ⇑e, inj'  …
    hι : ∀ {σ : Equiv.Perm n} {x : n}, Eq (ι σ x) (σ x)
    ⊢ Eq (HAdd.hAdd (Matrix.detp 1 (HMul.hMul A B)) (HAdd.hAdd (HMul.hMul (Matrix. …
  -/
  let bij : Finset (n → n) := (disjUnion (ofSign 1) (ofSign (-1)) ofSign_disjoint).map ι
  replace h (s) : detp s (A * B) =
      ∑ σ ∈ bijᶜ, ∑ τ ∈ ofSign s, ∏ i : n, A i (σ i) * B (σ i) (τ i) +
        (detp 1 A * detp s B + detp (-1) A * detp (-s) B) := by
    simp_rw [h, neg_mul_neg, mul_one, detp, mul_apply, prod_univ_sum, Fintype.piFinset_univ]
    rw [sum_comm, ← sum_compl_add_sum bij, sum_map, sum_disjUnion]
    simp_rw [hι]
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hf : ∀ {s t : Units Int} {σ : Equiv.Perm n}, Membership.mem (Equiv.Perm.ofSign …
    ι : Function.Embedding (Equiv.Perm n) (n → n) := { toFun := fun e => ⇑e, inj'  …
    hι : ∀ {σ : Equiv.Perm n} {x : n}, Eq (ι σ x) (σ x)
    bij : Finset (n → n) := Finset.map ι ((Equiv.Perm.ofSign 1).disjUnion (Equiv.P …
    h : ∀ (s : Units Int), Eq (Matrix.detp s (HMul.hMul A B)) (HAdd.hAdd ((HasComp …
    ⊢ Eq (HAdd.hAdd (Matrix.detp 1 (HMul.hMul A B)) (HAdd.hAdd (HMul.hMul (Matrix. …
  -/
  rw [h, h, neg_neg, add_assoc]
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hf : ∀ {s t : Units Int} {σ : Equiv.Perm n}, Membership.mem (Equiv.Perm.ofSign …
    ι : Function.Embedding (Equiv.Perm n) (n → n) := { toFun := fun e => ⇑e, inj'  …
    hι : ∀ {σ : Equiv.Perm n} {x : n}, Eq (ι σ x) (σ x)
    bij : Finset (n → n) := Finset.map ι ((Equiv.Perm.ofSign 1).disjUnion (Equiv.P …
    h : ∀ (s : Units Int), Eq (Matrix.detp s (HMul.hMul A B)) (HAdd.hAdd ((HasComp …
    ⊢ Eq (HAdd.hAdd ((HasCompl.compl bij).sum fun σ => (Equiv.Perm.ofSign 1).sum f …
  -/
  conv_rhs => rw [add_assoc]
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hf : ∀ {s t : Units Int} {σ : Equiv.Perm n}, Membership.mem (Equiv.Perm.ofSign …
    ι : Function.Embedding (Equiv.Perm n) (n → n) := { toFun := fun e => ⇑e, inj'  …
    hι : ∀ {σ : Equiv.Perm n} {x : n}, Eq (ι σ x) (σ x)
    bij : Finset (n → n) := Finset.map ι ((Equiv.Perm.ofSign 1).disjUnion (Equiv.P …
    h : ∀ (s : Units Int), Eq (Matrix.detp s (HMul.hMul A B)) (HAdd.hAdd ((HasComp …
    ⊢ Eq (HAdd.hAdd ((HasCompl.compl bij).sum fun σ => (Equiv.Perm.ofSign 1).sum f …
  -/
  refine congr_arg₂ (· + ·) (sum_congr rfl fun σ hσ ↦ ?_) (add_comm _ _)
  replace hσ : ¬ Function.Injective σ := by
    contrapose! hσ
    rw [not_mem_compl, mem_map, ofSign_disjUnion]
    exact ⟨Equiv.ofBijective σ hσ.bijective_of_finite, mem_univ _, rfl⟩
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hf : ∀ {s t : Units Int} {σ : Equiv.Perm n}, Membership.mem (Equiv.Perm.ofSign …
    ι : Function.Embedding (Equiv.Perm n) (n → n) := { toFun := fun e => ⇑e, inj'  …
    hι : ∀ {σ : Equiv.Perm n} {x : n}, Eq (ι σ x) (σ x)
    bij : Finset (n → n) := Finset.map ι ((Equiv.Perm.ofSign 1).disjUnion (Equiv.P …
    h : ∀ (s : Units Int), Eq (Matrix.detp s (HMul.hMul A B)) (HAdd.hAdd ((HasComp …
    σ : n → n
    hσ : Not (Function.Injective σ)
    ⊢ Eq ((Equiv.Perm.ofSign 1).sum fun τ => Finset.univ.prod fun i => HMul.hMul ( …
  -/
  obtain ⟨i, j, hσ, hij⟩ := Function.not_injective_iff.mp hσ
  replace hσ k : σ (swap i j k) = σ k := by
    rw [swap_apply_def]
    split_ifs with h h <;> simp only [hσ, h]
  /-
    case intro.intro.intro
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hf : ∀ {s t : Units Int} {σ : Equiv.Perm n}, Membership.mem (Equiv.Perm.ofSign …
    ι : Function.Embedding (Equiv.Perm n) (n → n) := { toFun := fun e => ⇑e, inj'  …
    hι : ∀ {σ : Equiv.Perm n} {x : n}, Eq (ι σ x) (σ x)
    bij : Finset (n → n) := Finset.map ι ((Equiv.Perm.ofSign 1).disjUnion (Equiv.P …
    h : ∀ (s : Units Int), Eq (Matrix.detp s (HMul.hMul A B)) (HAdd.hAdd ((HasComp …
    σ : n → n
    hσ✝ : Not (Function.Injective σ)
    i j : n
    hij : Ne i j
    hσ : ∀ (k : n), Eq (σ ((Equiv.swap i j) k)) (σ k)
    ⊢ Eq ((Equiv.Perm.ofSign 1).sum fun τ => Finset.univ.prod fun i => HMul.hMul ( …
  -/
  rw [← mul_neg_one, hf (mem_ofSign.mpr (sign_swap hij)), sum_map]
  /-
    case intro.intro.intro
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hf : ∀ {s t : Units Int} {σ : Equiv.Perm n}, Membership.mem (Equiv.Perm.ofSign …
    ι : Function.Embedding (Equiv.Perm n) (n → n) := { toFun := fun e => ⇑e, inj'  …
    hι : ∀ {σ : Equiv.Perm n} {x : n}, Eq (ι σ x) (σ x)
    bij : Finset (n → n) := Finset.map ι ((Equiv.Perm.ofSign 1).disjUnion (Equiv.P …
    h : ∀ (s : Units Int), Eq (Matrix.detp s (HMul.hMul A B)) (HAdd.hAdd ((HasComp …
    σ : n → n
    hσ✝ : Not (Function.Injective σ)
    i j : n
    hij : Ne i j
    hσ : ∀ (k : n), Eq (σ ((Equiv.swap i j) k)) (σ k)
    ⊢ Eq ((Equiv.Perm.ofSign 1).sum fun τ => Finset.univ.prod fun i => HMul.hMul ( …
  -/
  simp_rw [prod_mul_distrib, mulRightEmbedding_apply, Perm.mul_apply]
  /-
    case intro.intro.intro
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hf : ∀ {s t : Units Int} {σ : Equiv.Perm n}, Membership.mem (Equiv.Perm.ofSign …
    ι : Function.Embedding (Equiv.Perm n) (n → n) := { toFun := fun e => ⇑e, inj'  …
    hι : ∀ {σ : Equiv.Perm n} {x : n}, Eq (ι σ x) (σ x)
    bij : Finset (n → n) := Finset.map ι ((Equiv.Perm.ofSign 1).disjUnion (Equiv.P …
    h : ∀ (s : Units Int), Eq (Matrix.detp s (HMul.hMul A B)) (HAdd.hAdd ((HasComp …
    σ : n → n
    hσ✝ : Not (Function.Injective σ)
    i j : n
    hij : Ne i j
    hσ : ∀ (k : n), Eq (σ ((Equiv.swap i j) k)) (σ k)
    ⊢ Eq ((Equiv.Perm.ofSign 1).sum fun x => HMul.hMul (Finset.univ.prod fun x =>  …
  -/
  refine sum_congr rfl fun τ hτ ↦ congr_arg (_ *  ·) ?_
  /-
    case intro.intro.intro
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hf : ∀ {s t : Units Int} {σ : Equiv.Perm n}, Membership.mem (Equiv.Perm.ofSign …
    ι : Function.Embedding (Equiv.Perm n) (n → n) := { toFun := fun e => ⇑e, inj'  …
    hι : ∀ {σ : Equiv.Perm n} {x : n}, Eq (ι σ x) (σ x)
    bij : Finset (n → n) := Finset.map ι ((Equiv.Perm.ofSign 1).disjUnion (Equiv.P …
    h : ∀ (s : Units Int), Eq (Matrix.detp s (HMul.hMul A B)) (HAdd.hAdd ((HasComp …
    σ : n → n
    hσ✝ : Not (Function.Injective σ)
    i j : n
    hij : Ne i j
    hσ : ∀ (k : n), Eq (σ ((Equiv.swap i j) k)) (σ k)
    τ : Equiv.Perm n
    hτ : Membership.mem (Equiv.Perm.ofSign 1) τ
    ⊢ Eq (Finset.univ.prod fun x => B (σ x) (τ x)) (Finset.univ.prod fun x => B (σ …
  -/
  rw [← Equiv.prod_comp (swap i j)]
  /-
    case intro.intro.intro
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hf : ∀ {s t : Units Int} {σ : Equiv.Perm n}, Membership.mem (Equiv.Perm.ofSign …
    ι : Function.Embedding (Equiv.Perm n) (n → n) := { toFun := fun e => ⇑e, inj'  …
    hι : ∀ {σ : Equiv.Perm n} {x : n}, Eq (ι σ x) (σ x)
    bij : Finset (n → n) := Finset.map ι ((Equiv.Perm.ofSign 1).disjUnion (Equiv.P …
    h : ∀ (s : Units Int), Eq (Matrix.detp s (HMul.hMul A B)) (HAdd.hAdd ((HasComp …
    σ : n → n
    hσ✝ : Not (Function.Injective σ)
    i j : n
    hij : Ne i j
    hσ : ∀ (k : n), Eq (σ ((Equiv.swap i j) k)) (σ k)
    τ : Equiv.Perm n
    hτ : Membership.mem (Equiv.Perm.ofSign 1) τ
    ⊢ Eq (Finset.univ.prod fun i_1 => B (σ ((Equiv.swap i j) i_1)) (τ ((Equiv.swap …
  -/
  simp only [hσ]
  /-
    🎉 no goals
  -/


theorem mul_adjp_apply_eq : (A * adjp s A) i i = detp s A := by
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    s : Units Int
    A : Matrix n n R
    i : n
    ⊢ Eq (HMul.hMul A (Matrix.adjp s A) i i) (Matrix.detp s A)
  -/
  have key := sum_fiberwise_eq_sum_filter (ofSign s) univ (· i) fun σ ↦ ∏ k, A k (σ k)
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    s : Units Int
    A : Matrix n n R
    i : n
    key : Eq (Finset.univ.sum fun j => (Finset.filter (fun i_1 => Eq (i_1 i) j) (E …
    ⊢ Eq (HMul.hMul A (Matrix.adjp s A) i i) (Matrix.detp s A)
  -/
  simp_rw [mem_univ, filter_True] at key
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    s : Units Int
    A : Matrix n n R
    i : n
    key : Eq (Finset.univ.sum fun j => (Finset.filter (fun i_1 => Eq (i_1 i) j) (E …
    ⊢ Eq (HMul.hMul A (Matrix.adjp s A) i i) (Matrix.detp s A)
  -/
  simp_rw [mul_apply, adjp_apply, mul_sum, detp, ← key]
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    s : Units Int
    A : Matrix n n R
    i : n
    key : Eq (Finset.univ.sum fun j => (Finset.filter (fun i_1 => Eq (i_1 i) j) (E …
    ⊢ Eq (Finset.univ.sum fun x => (Finset.filter (fun x_1 => Eq (x_1 i) x) (Equiv …
  -/
  refine sum_congr rfl fun x hx ↦ sum_congr rfl fun σ hσ ↦ ?_
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    s : Units Int
    A : Matrix n n R
    i : n
    key : Eq (Finset.univ.sum fun j => (Finset.filter (fun i_1 => Eq (i_1 i) j) (E …
    x : n
    hx : Membership.mem Finset.univ x
    σ : Equiv.Perm n
    hσ : Membership.mem (Finset.filter (fun i_1 => Eq (i_1 i) x) (Equiv.Perm.ofSig …
    ⊢ Eq (HMul.hMul (A i x) ((HasCompl.compl (Singleton.singleton i)).prod fun k = …
  -/
  rw [← prod_mul_prod_compl ({i} : Finset n), prod_singleton, (mem_filter.mp hσ).2]
  /-
    🎉 no goals
  -/


theorem mul_adjp_apply_ne (h : i ≠ j) : (A * adjp 1 A) i j = (A * adjp (-1) A) i j := by
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A : Matrix n n R
    i j : n
    h : Ne i j
    ⊢ Eq (HMul.hMul A (Matrix.adjp 1 A) i j) (HMul.hMul A (Matrix.adjp (-1) A) i j)
  -/
  simp_rw [mul_apply, adjp_apply, mul_sum, sum_sigma']
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A : Matrix n n R
    i j : n
    h : Ne i j
    ⊢ Eq ((Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x j) a) (Equiv.P …
  -/
  let f : (Σ x : n, Perm n) → (Σ x : n, Perm n) := fun ⟨x, σ⟩ ↦ ⟨σ i, σ * swap i j⟩
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A : Matrix n n R
    i j : n
    h : Ne i j
    f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
    ⊢ Eq ((Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x j) a) (Equiv.P …
  -/
  let t s : Finset (Σ x : n, Perm n) := univ.sigma fun x ↦ (ofSign s).filter fun σ ↦ σ j = x
  have hf {s} : ∀ p ∈ t s, f (f p) = p := by
    intro ⟨x, σ⟩ hp
    rw [mem_sigma, mem_filter, mem_ofSign] at hp
    simp_rw [f, Perm.mul_apply, swap_apply_left, hp.2.2, mul_swap_mul_self]
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A : Matrix n n R
    i j : n
    h : Ne i j
    f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
    t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
    hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
    ⊢ Eq ((Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x j) a) (Equiv.P …
  -/
  refine sum_bij' (fun p _ ↦ f p) (fun p _ ↦ f p) ?_ ?_ hf hf ?_
    /-
      case refine_1
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
      t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
      hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
      ⊢ ∀ (a : Sigma fun i => Equiv.Perm n) (ha : Membership.mem (Finset.univ.sigma  …
    -/
  · intro ⟨x, σ⟩ hp
    /-
      case refine_1
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
      t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
      hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
      x : n
      σ : Equiv.Perm n
      hp : Membership.mem (Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x  …
      ⊢ Membership.mem (Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x j)  …
    -/
    rw [mem_sigma, mem_filter, mem_ofSign] at hp ⊢
    /-
      case refine_1
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
      t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
      hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
      x : n
      σ : Equiv.Perm n
      hp✝ : Membership.mem (Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x …
      hp : And (Membership.mem Finset.univ ⟨x, σ⟩.fst) (And (Eq (Equiv.Perm.sign ⟨x, …
      ⊢ And (Membership.mem Finset.univ ((fun p x => f p) ⟨x, σ⟩ hp✝).fst) (And (Eq  …
    -/
    rw [Perm.mul_apply, sign_mul, hp.2.1, sign_swap h, swap_apply_right]
    /-
      case refine_1
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
      t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
      hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
      x : n
      σ : Equiv.Perm n
      hp✝ : Membership.mem (Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x …
      hp : And (Membership.mem Finset.univ ⟨x, σ⟩.fst) (And (Eq (Equiv.Perm.sign ⟨x, …
      ⊢ And (Membership.mem Finset.univ ((fun p x => f p) ⟨x, σ⟩ hp✝).fst) (And (Eq  …
    -/
    exact ⟨mem_univ (σ i), rfl, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
      t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
      hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
      ⊢ ∀ (a : Sigma fun i => Equiv.Perm n) (ha : Membership.mem (Finset.univ.sigma  …
    -/
  · intro ⟨x, σ⟩ hp
    /-
      case refine_2
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
      t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
      hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
      x : n
      σ : Equiv.Perm n
      hp : Membership.mem (Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x  …
      ⊢ Membership.mem (Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x j)  …
    -/
    rw [mem_sigma, mem_filter, mem_ofSign] at hp ⊢
    /-
      case refine_2
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
      t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
      hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
      x : n
      σ : Equiv.Perm n
      hp✝ : Membership.mem (Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x …
      hp : And (Membership.mem Finset.univ ⟨x, σ⟩.fst) (And (Eq (Equiv.Perm.sign ⟨x, …
      ⊢ And (Membership.mem Finset.univ ((fun p x => f p) ⟨x, σ⟩ hp✝).fst) (And (Eq  …
    -/
    rw [Perm.mul_apply, sign_mul, hp.2.1, sign_swap h, swap_apply_right]
    /-
      case refine_2
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
      t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
      hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
      x : n
      σ : Equiv.Perm n
      hp✝ : Membership.mem (Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x …
      hp : And (Membership.mem Finset.univ ⟨x, σ⟩.fst) (And (Eq (Equiv.Perm.sign ⟨x, …
      ⊢ And (Membership.mem Finset.univ ((fun p x => f p) ⟨x, σ⟩ hp✝).fst) (And (Eq  …
    -/
    exact ⟨mem_univ (σ i), rfl, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
      t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
      hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
      ⊢ ∀ (a : Sigma fun i => Equiv.Perm n) (ha : Membership.mem (Finset.univ.sigma  …
    -/
  · intro ⟨x, σ⟩ hp
    /-
      case refine_3
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
      t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
      hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
      x : n
      σ : Equiv.Perm n
      hp : Membership.mem (Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x  …
      ⊢ Eq (HMul.hMul (A i ⟨x, σ⟩.fst) ((HasCompl.compl (Singleton.singleton j)).pro …
    -/
    rw [mem_sigma, mem_filter, mem_ofSign] at hp
    have key : ({j}ᶜ : Finset n) = disjUnion ({i} : Finset n) ({i, j} : Finset n)ᶜ (by simp) := by
      rw [singleton_disjUnion, cons_eq_insert, compl_insert, insert_erase]
      rwa [mem_compl, mem_singleton]
    /-
      case refine_3
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
      t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
      hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
      x : n
      σ : Equiv.Perm n
      hp✝ : Membership.mem (Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x …
      hp : And (Membership.mem Finset.univ ⟨x, σ⟩.fst) (And (Eq (Equiv.Perm.sign ⟨x, …
      key : Eq (HasCompl.compl (Singleton.singleton j)) ((Singleton.singleton i).dis …
      ⊢ Eq (HMul.hMul (A i ⟨x, σ⟩.fst) ((HasCompl.compl (Singleton.singleton j)).pro …
    -/
    simp_rw [key, prod_disjUnion, prod_singleton, Perm.mul_apply, swap_apply_left, ← mul_assoc]
    /-
      case refine_3
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
      t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
      hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
      x : n
      σ : Equiv.Perm n
      hp✝ : Membership.mem (Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x …
      hp : And (Membership.mem Finset.univ ⟨x, σ⟩.fst) (And (Eq (Equiv.Perm.sign ⟨x, …
      key : Eq (HasCompl.compl (Singleton.singleton j)) ((Singleton.singleton i).dis …
      ⊢ Eq (HMul.hMul (HMul.hMul (A i x) (A i (σ i))) ((HasCompl.compl (Insert.inser …
    -/
    rw [mul_comm (A i x) (A i (σ i)), hp.2.2]
    /-
      case refine_3
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
      t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
      hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
      x : n
      σ : Equiv.Perm n
      hp✝ : Membership.mem (Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x …
      hp : And (Membership.mem Finset.univ ⟨x, σ⟩.fst) (And (Eq (Equiv.Perm.sign ⟨x, …
      key : Eq (HasCompl.compl (Singleton.singleton j)) ((Singleton.singleton i).dis …
      ⊢ Eq (HMul.hMul (HMul.hMul (A i (σ i)) (A i x)) ((HasCompl.compl (Insert.inser …
    -/
    refine congr_arg _ (prod_congr rfl fun x hx ↦ ?_)
    /-
      case refine_3
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
      t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
      hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
      x✝ : n
      σ : Equiv.Perm n
      hp✝ : Membership.mem (Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x …
      hp : And (Membership.mem Finset.univ ⟨x✝, σ⟩.fst) (And (Eq (Equiv.Perm.sign ⟨x …
      key : Eq (HasCompl.compl (Singleton.singleton j)) ((Singleton.singleton i).dis …
      x : n
      hx : Membership.mem (HasCompl.compl (Insert.insert i (Singleton.singleton j))) x
      ⊢ Eq (A x (σ x)) (A x (σ ((Equiv.swap i j) x)))
    -/
    rw [mem_compl, mem_insert, mem_singleton, not_or] at hx
    /-
      case refine_3
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      f : (Sigma fun x => Equiv.Perm n) → Sigma fun x => Equiv.Perm n := fun x => Ma …
      t : Units Int → Finset (Sigma fun x => Equiv.Perm n) := fun s => Finset.univ.s …
      hf : ∀ {s : Units Int} (p : Sigma fun x => Equiv.Perm n), Membership.mem (t s) …
      x✝ : n
      σ : Equiv.Perm n
      hp✝ : Membership.mem (Finset.univ.sigma fun a => Finset.filter (fun x => Eq (x …
      hp : And (Membership.mem Finset.univ ⟨x✝, σ⟩.fst) (And (Eq (Equiv.Perm.sign ⟨x …
      key : Eq (HasCompl.compl (Singleton.singleton j)) ((Singleton.singleton i).dis …
      x : n
      hx : And (Not (Eq x i)) (Not (Eq x j))
      ⊢ Eq (A x (σ x)) (A x (σ ((Equiv.swap i j) x)))
    -/
    rw [swap_apply_of_ne_of_ne hx.1 hx.2]
    /-
      🎉 no goals
    -/


theorem mul_adjp_add_detp : A * adjp 1 A + detp (-1) A • 1 = A * adjp (-1) A + detp 1 A • 1 := by
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A : Matrix n n R
    ⊢ Eq (HAdd.hAdd (HMul.hMul A (Matrix.adjp 1 A)) (HSMul.hSMul (Matrix.detp (-1) …
  -/
  ext i j
  /-
    case a
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A : Matrix n n R
    i j : n
    ⊢ Eq (HAdd.hAdd (HMul.hMul A (Matrix.adjp 1 A)) (HSMul.hSMul (Matrix.detp (-1) …
  -/
  rcases eq_or_ne i j with rfl | h <;> simp_rw [add_apply, smul_apply, smul_eq_mul]
    /-
      case a.inl
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i : n
      ⊢ Eq (HAdd.hAdd (HMul.hMul A (Matrix.adjp 1 A) i i) (HMul.hMul (Matrix.detp (- …
    -/
  · simp_rw [mul_adjp_apply_eq, one_apply_eq, mul_one, add_comm]
    /-
      🎉 no goals
    -/
    /-
      case a.inr
      n : Type u_1
      R : Type u_3
      inst✝² : Fintype n
      inst✝¹ : DecidableEq n
      inst✝ : CommSemiring R
      A : Matrix n n R
      i j : n
      h : Ne i j
      ⊢ Eq (HAdd.hAdd (HMul.hMul A (Matrix.adjp 1 A) i j) (HMul.hMul (Matrix.detp (- …
    -/
  · simp_rw [mul_adjp_apply_ne A i j h, one_apply_ne h, mul_zero]
    /-
      🎉 no goals
    -/


theorem isAddUnit_mul (hAB : A * B = 1) (i j k : n) (hij : i ≠ j) : IsAddUnit (A i k * B k j) := by
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    i j k : n
    hij : Ne i j
    ⊢ IsAddUnit (HMul.hMul (A i k) (B k j))
  -/
  revert k
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    i j : n
    hij : Ne i j
    ⊢ ∀ (k : n), IsAddUnit (HMul.hMul (A i k) (B k j))
  -/
  rw [← IsAddUnit.sum_univ_iff, ← mul_apply, hAB, one_apply_ne hij]
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    i j : n
    hij : Ne i j
    ⊢ IsAddUnit 0
  -/
  exact isAddUnit_zero
  /-
    🎉 no goals
  -/


theorem isAddUnit_detp_mul_detp (hAB : A * B = 1) :
    IsAddUnit (detp 1 A * detp (-1) B + detp (-1) A * detp 1 B) := by
  suffices h : ∀ {s t}, s ≠ t → IsAddUnit (detp s A * detp t B) from
    (h (by decide)).add (h (by decide))
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    ⊢ ∀ {s t : Units Int}, Ne s t → IsAddUnit (HMul.hMul (Matrix.detp s A) (Matrix …
  -/
  intro s t h
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    h : Ne s t
    ⊢ IsAddUnit (HMul.hMul (Matrix.detp s A) (Matrix.detp t B))
  -/
  simp_rw [detp, sum_mul_sum, IsAddUnit.sum_iff]
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    h : Ne s t
    ⊢ ∀ (a : Equiv.Perm n), Membership.mem (Equiv.Perm.ofSign s) a → ∀ (a_2 : Equi …
  -/
  intro σ hσ τ hτ
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    h : Ne s t
    σ : Equiv.Perm n
    hσ : Membership.mem (Equiv.Perm.ofSign s) σ
    τ : Equiv.Perm n
    hτ : Membership.mem (Equiv.Perm.ofSign t) τ
    ⊢ IsAddUnit (HMul.hMul (Finset.univ.prod fun k => A k (σ k)) (Finset.univ.prod …
  -/
  rw [mem_ofSign] at hσ hτ
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    h : Ne s t
    σ : Equiv.Perm n
    hσ : Eq (Equiv.Perm.sign σ) s
    τ : Equiv.Perm n
    hτ : Eq (Equiv.Perm.sign τ) t
    ⊢ IsAddUnit (HMul.hMul (Finset.univ.prod fun k => A k (σ k)) (Finset.univ.prod …
  -/
  rw [← hσ, ← hτ, ← sign_inv] at h
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    σ : Equiv.Perm n
    hσ : Eq (Equiv.Perm.sign σ) s
    τ : Equiv.Perm n
    h : Ne (Equiv.Perm.sign (Inv.inv σ)) (Equiv.Perm.sign τ)
    hτ : Eq (Equiv.Perm.sign τ) t
    ⊢ IsAddUnit (HMul.hMul (Finset.univ.prod fun k => A k (σ k)) (Finset.univ.prod …
  -/
  replace h := ne_of_apply_ne sign h
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    σ : Equiv.Perm n
    hσ : Eq (Equiv.Perm.sign σ) s
    τ : Equiv.Perm n
    hτ : Eq (Equiv.Perm.sign τ) t
    h : Ne (Inv.inv σ) τ
    ⊢ IsAddUnit (HMul.hMul (Finset.univ.prod fun k => A k (σ k)) (Finset.univ.prod …
  -/
  rw [ne_eq, eq_comm, eq_inv_iff_mul_eq_one, eq_comm] at h
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    σ : Equiv.Perm n
    hσ : Eq (Equiv.Perm.sign σ) s
    τ : Equiv.Perm n
    hτ : Eq (Equiv.Perm.sign τ) t
    h : Not (Eq 1 (HMul.hMul τ σ))
    ⊢ IsAddUnit (HMul.hMul (Finset.univ.prod fun k => A k (σ k)) (Finset.univ.prod …
  -/
  simp_rw [Equiv.ext_iff, not_forall, Perm.mul_apply, Perm.one_apply] at h
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    σ : Equiv.Perm n
    hσ : Eq (Equiv.Perm.sign σ) s
    τ : Equiv.Perm n
    hτ : Eq (Equiv.Perm.sign τ) t
    h : Exists fun x => Not (Eq x (τ (σ x)))
    ⊢ IsAddUnit (HMul.hMul (Finset.univ.prod fun k => A k (σ k)) (Finset.univ.prod …
  -/
  obtain ⟨k, hk⟩ := h
  rw [mul_comm, ← Equiv.prod_comp σ, mul_comm, ← prod_mul_distrib,
    ← mul_prod_erase univ _ (mem_univ k), ← smul_eq_mul]
  /-
    case intro
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    σ : Equiv.Perm n
    hσ : Eq (Equiv.Perm.sign σ) s
    τ : Equiv.Perm n
    hτ : Eq (Equiv.Perm.sign τ) t
    k : n
    hk : Not (Eq k (τ (σ k)))
    ⊢ IsAddUnit (HSMul.hSMul (HMul.hMul (A k (σ k)) (B (σ k) (τ (σ k)))) ((Finset. …
  -/
  exact (isAddUnit_mul hAB k (τ (σ k)) (σ k) hk).smul_right _
  /-
    🎉 no goals
  -/


theorem isAddUnit_detp_smul_mul_adjp (hAB : A * B = 1) :
    IsAddUnit (detp 1 A • (B * adjp (-1) B) + detp (-1) A • (B * adjp 1 B)) := by
  suffices h : ∀ {s t}, s ≠ t → IsAddUnit (detp s A • (B * adjp t B)) from
    (h (by decide)).add (h (by decide))
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    ⊢ ∀ {s t : Units Int}, Ne s t → IsAddUnit (HSMul.hSMul (Matrix.detp s A) (HMul …
  -/
  intro s t h
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    h : Ne s t
    ⊢ IsAddUnit (HSMul.hSMul (Matrix.detp s A) (HMul.hMul B (Matrix.adjp t B)))
  -/
  rw [isAddUnit_iff]
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    h : Ne s t
    ⊢ ∀ (i j : n), IsAddUnit (HSMul.hSMul (Matrix.detp s A) (HMul.hMul B (Matrix.a …
  -/
  intro i j
  simp_rw [smul_apply, smul_eq_mul, mul_apply, detp, adjp_apply, mul_sum, sum_mul,
    IsAddUnit.sum_iff]
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    h : Ne s t
    i j : n
    ⊢ ∀ (a : n), Membership.mem Finset.univ a → ∀ (a_2 : Equiv.Perm n), Membership …
  -/
  intro k hk σ hσ τ hτ
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    h : Ne s t
    i j k : n
    hk : Membership.mem Finset.univ k
    σ : Equiv.Perm n
    hσ : Membership.mem (Finset.filter (fun x => Eq (x j) k) (Equiv.Perm.ofSign t) …
    τ : Equiv.Perm n
    hτ : Membership.mem (Equiv.Perm.ofSign s) τ
    ⊢ IsAddUnit (HMul.hMul (Finset.univ.prod fun k => A k (τ k)) (HMul.hMul (B i k …
  -/
  rw [mem_filter] at hσ
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    h : Ne s t
    i j k : n
    hk : Membership.mem Finset.univ k
    σ : Equiv.Perm n
    hσ : And (Membership.mem (Equiv.Perm.ofSign t) σ) (Eq (σ j) k)
    τ : Equiv.Perm n
    hτ : Membership.mem (Equiv.Perm.ofSign s) τ
    ⊢ IsAddUnit (HMul.hMul (Finset.univ.prod fun k => A k (τ k)) (HMul.hMul (B i k …
  -/
  rw [mem_ofSign] at hσ hτ
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    h : Ne s t
    i j k : n
    hk : Membership.mem Finset.univ k
    σ : Equiv.Perm n
    hσ : And (Eq (Equiv.Perm.sign σ) t) (Eq (σ j) k)
    τ : Equiv.Perm n
    hτ : Eq (Equiv.Perm.sign τ) s
    ⊢ IsAddUnit (HMul.hMul (Finset.univ.prod fun k => A k (τ k)) (HMul.hMul (B i k …
  -/
  rw [← hσ.1, ← hτ, ← sign_inv] at h
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    i j k : n
    hk : Membership.mem Finset.univ k
    σ : Equiv.Perm n
    hσ : And (Eq (Equiv.Perm.sign σ) t) (Eq (σ j) k)
    τ : Equiv.Perm n
    h : Ne (Equiv.Perm.sign (Inv.inv τ)) (Equiv.Perm.sign σ)
    hτ : Eq (Equiv.Perm.sign τ) s
    ⊢ IsAddUnit (HMul.hMul (Finset.univ.prod fun k => A k (τ k)) (HMul.hMul (B i k …
  -/
  replace h := ne_of_apply_ne sign h
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    i j k : n
    hk : Membership.mem Finset.univ k
    σ : Equiv.Perm n
    hσ : And (Eq (Equiv.Perm.sign σ) t) (Eq (σ j) k)
    τ : Equiv.Perm n
    hτ : Eq (Equiv.Perm.sign τ) s
    h : Ne (Inv.inv τ) σ
    ⊢ IsAddUnit (HMul.hMul (Finset.univ.prod fun k => A k (τ k)) (HMul.hMul (B i k …
  -/
  rw [ne_eq, eq_comm, eq_inv_iff_mul_eq_one] at h
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    i j k : n
    hk : Membership.mem Finset.univ k
    σ : Equiv.Perm n
    hσ : And (Eq (Equiv.Perm.sign σ) t) (Eq (σ j) k)
    τ : Equiv.Perm n
    hτ : Eq (Equiv.Perm.sign τ) s
    h : Not (Eq (HMul.hMul σ τ) 1)
    ⊢ IsAddUnit (HMul.hMul (Finset.univ.prod fun k => A k (τ k)) (HMul.hMul (B i k …
  -/
  obtain ⟨l, hl1, hl2⟩ := exists_ne_of_one_lt_card (one_lt_card_support_of_ne_one h) (τ⁻¹ j)
  /-
    case intro.intro
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    i j k : n
    hk : Membership.mem Finset.univ k
    σ : Equiv.Perm n
    hσ : And (Eq (Equiv.Perm.sign σ) t) (Eq (σ j) k)
    τ : Equiv.Perm n
    hτ : Eq (Equiv.Perm.sign τ) s
    h : Not (Eq (HMul.hMul σ τ) 1)
    l : n
    hl1 : Membership.mem (HMul.hMul σ τ).support l
    hl2 : Ne l ((Inv.inv τ) j)
    ⊢ IsAddUnit (HMul.hMul (Finset.univ.prod fun k => A k (τ k)) (HMul.hMul (B i k …
  -/
  rw [mem_support, ne_comm] at hl1
  /-
    case intro.intro
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    i j k : n
    hk : Membership.mem Finset.univ k
    σ : Equiv.Perm n
    hσ : And (Eq (Equiv.Perm.sign σ) t) (Eq (σ j) k)
    τ : Equiv.Perm n
    hτ : Eq (Equiv.Perm.sign τ) s
    h : Not (Eq (HMul.hMul σ τ) 1)
    l : n
    hl1 : Ne l ((HMul.hMul σ τ) l)
    hl2 : Ne l ((Inv.inv τ) j)
    ⊢ IsAddUnit (HMul.hMul (Finset.univ.prod fun k => A k (τ k)) (HMul.hMul (B i k …
  -/
  rw [ne_eq, ← mem_singleton, ← mem_compl] at hl2
  /-
    case intro.intro
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    i j k : n
    hk : Membership.mem Finset.univ k
    σ : Equiv.Perm n
    hσ : And (Eq (Equiv.Perm.sign σ) t) (Eq (σ j) k)
    τ : Equiv.Perm n
    hτ : Eq (Equiv.Perm.sign τ) s
    h : Not (Eq (HMul.hMul σ τ) 1)
    l : n
    hl1 : Ne l ((HMul.hMul σ τ) l)
    hl2 : Membership.mem (HasCompl.compl (Singleton.singleton ((Inv.inv τ) j))) l
    ⊢ IsAddUnit (HMul.hMul (Finset.univ.prod fun k => A k (τ k)) (HMul.hMul (B i k …
  -/
  rw [← prod_mul_prod_compl {τ⁻¹ j}, mul_mul_mul_comm, mul_comm, ← smul_eq_mul]
  /-
    case intro.intro
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    i j k : n
    hk : Membership.mem Finset.univ k
    σ : Equiv.Perm n
    hσ : And (Eq (Equiv.Perm.sign σ) t) (Eq (σ j) k)
    τ : Equiv.Perm n
    hτ : Eq (Equiv.Perm.sign τ) s
    h : Not (Eq (HMul.hMul σ τ) 1)
    l : n
    hl1 : Ne l ((HMul.hMul σ τ) l)
    hl2 : Membership.mem (HasCompl.compl (Singleton.singleton ((Inv.inv τ) j))) l
    ⊢ IsAddUnit (HSMul.hSMul (HMul.hMul ((HasCompl.compl (Singleton.singleton ((In …
  -/
  apply IsAddUnit.smul_right
  have h0 : ∀ k, k ∈ ({τ⁻¹ j} : Finset n)ᶜ ↔ τ k ∈ ({j} : Finset n)ᶜ := by
    simp [inv_def, eq_symm_apply]
  /-
    case intro.intro.hr
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    i j k : n
    hk : Membership.mem Finset.univ k
    σ : Equiv.Perm n
    hσ : And (Eq (Equiv.Perm.sign σ) t) (Eq (σ j) k)
    τ : Equiv.Perm n
    hτ : Eq (Equiv.Perm.sign τ) s
    h : Not (Eq (HMul.hMul σ τ) 1)
    l : n
    hl1 : Ne l ((HMul.hMul σ τ) l)
    hl2 : Membership.mem (HasCompl.compl (Singleton.singleton ((Inv.inv τ) j))) l
    h0 : ∀ (k : n), Iff (Membership.mem (HasCompl.compl (Singleton.singleton ((Inv …
    ⊢ IsAddUnit (HMul.hMul ((HasCompl.compl (Singleton.singleton ((Inv.inv τ) j))) …
  -/
  rw [← prod_equiv τ h0 fun _ _ ↦ rfl, ← prod_mul_distrib, ← mul_prod_erase _ _ hl2, ← smul_eq_mul]
  /-
    case intro.intro.hr
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    s t : Units Int
    i j k : n
    hk : Membership.mem Finset.univ k
    σ : Equiv.Perm n
    hσ : And (Eq (Equiv.Perm.sign σ) t) (Eq (σ j) k)
    τ : Equiv.Perm n
    hτ : Eq (Equiv.Perm.sign τ) s
    h : Not (Eq (HMul.hMul σ τ) 1)
    l : n
    hl1 : Ne l ((HMul.hMul σ τ) l)
    hl2 : Membership.mem (HasCompl.compl (Singleton.singleton ((Inv.inv τ) j))) l
    h0 : ∀ (k : n), Iff (Membership.mem (HasCompl.compl (Singleton.singleton ((Inv …
    ⊢ IsAddUnit (HSMul.hSMul (HMul.hMul (A l (τ l)) (B (τ l) (σ (τ l)))) (((HasCom …
  -/
  exact (isAddUnit_mul hAB l (σ (τ l)) (τ l) hl1).smul_right _
  /-
    🎉 no goals
  -/


theorem detp_smul_add_adjp (hAB : A * B = 1) :
    detp 1 B • A + adjp (-1) B = detp (-1) B • A + adjp 1 B := by
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Matrix.detp 1 B) A) (Matrix.adjp (-1) B)) (HAdd. …
  -/
  have key := congr(A * $(mul_adjp_add_detp B))
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    key : Eq (HMul.hMul A (HAdd.hAdd (HMul.hMul B (Matrix.adjp 1 B)) (HSMul.hSMul  …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Matrix.detp 1 B) A) (Matrix.adjp (-1) B)) (HAdd. …
  -/
  simp_rw [mul_add, ← mul_assoc, hAB, one_mul, mul_smul, mul_one] at key
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    key : Eq (HAdd.hAdd (Matrix.adjp 1 B) (HSMul.hSMul (Matrix.detp (-1) B) A)) (H …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Matrix.detp 1 B) A) (Matrix.adjp (-1) B)) (HAdd. …
  -/
  rwa [add_comm, eq_comm, add_comm]
  /-
    🎉 no goals
  -/


theorem detp_smul_adjp (hAB : A * B = 1) :
    A + (detp 1 A • adjp (-1) B + detp (-1) A • adjp 1 B) =
      detp 1 A • adjp 1 B + detp (-1) A • adjp (-1) B := by
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    ⊢ Eq (HAdd.hAdd A (HAdd.hAdd (HSMul.hSMul (Matrix.detp 1 A) (Matrix.adjp (-1)  …
  -/
  have h0 := detp_mul A B
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    h0 : Eq (HAdd.hAdd (Matrix.detp 1 (HMul.hMul A B)) (HAdd.hAdd (HMul.hMul (Matr …
    ⊢ Eq (HAdd.hAdd A (HAdd.hAdd (HSMul.hSMul (Matrix.detp 1 A) (Matrix.adjp (-1)  …
  -/
  rw [hAB, detp_one_one, detp_neg_one_one, zero_add] at h0
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    h0 : Eq (HAdd.hAdd 1 (HAdd.hAdd (HMul.hMul (Matrix.detp 1 A) (Matrix.detp (-1) …
    ⊢ Eq (HAdd.hAdd A (HAdd.hAdd (HSMul.hSMul (Matrix.detp 1 A) (Matrix.adjp (-1)  …
  -/
  have h := detp_smul_add_adjp hAB
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    h0 : Eq (HAdd.hAdd 1 (HAdd.hAdd (HMul.hMul (Matrix.detp 1 A) (Matrix.detp (-1) …
    h : Eq (HAdd.hAdd (HSMul.hSMul (Matrix.detp 1 B) A) (Matrix.adjp (-1) B)) (HAd …
    ⊢ Eq (HAdd.hAdd A (HAdd.hAdd (HSMul.hSMul (Matrix.detp 1 A) (Matrix.adjp (-1)  …
  -/
  replace h := congr(detp 1 A • $h + detp (-1) A • $h.symm)
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    h0 : Eq (HAdd.hAdd 1 (HAdd.hAdd (HMul.hMul (Matrix.detp 1 A) (Matrix.detp (-1) …
    h : Eq (HAdd.hAdd (HSMul.hSMul (Matrix.detp 1 A) (HAdd.hAdd (HSMul.hSMul (Matr …
    ⊢ Eq (HAdd.hAdd A (HAdd.hAdd (HSMul.hSMul (Matrix.detp 1 A) (Matrix.adjp (-1)  …
  -/
  simp only [smul_add, smul_smul] at h
  rwa [add_add_add_comm, ← add_smul, add_add_add_comm, ← add_smul, ← h0, add_smul, one_smul,
    add_comm A, add_assoc, ((isAddUnit_detp_mul_detp hAB).smul_right _).add_right_inj] at h


theorem mul_eq_one_comm : A * B = 1 ↔ B * A = 1 := by
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    ⊢ Iff (Eq (HMul.hMul A B) 1) (Eq (HMul.hMul B A) 1)
  -/
  suffices h : ∀ A B : Matrix n n R, A * B = 1 → B * A = 1 from ⟨h A B, h B A⟩
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A B : Matrix n n R
    ⊢ ∀ (A B : Matrix n n R), Eq (HMul.hMul A B) 1 → Eq (HMul.hMul B A) 1
  -/
  intro A B hAB
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A✝ B✝ A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    ⊢ Eq (HMul.hMul B A) 1
  -/
  have h0 := detp_mul A B
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A✝ B✝ A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    h0 : Eq (HAdd.hAdd (Matrix.detp 1 (HMul.hMul A B)) (HAdd.hAdd (HMul.hMul (Matr …
    ⊢ Eq (HMul.hMul B A) 1
  -/
  rw [hAB, detp_one_one, detp_neg_one_one, zero_add] at h0
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A✝ B✝ A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    h0 : Eq (HAdd.hAdd 1 (HAdd.hAdd (HMul.hMul (Matrix.detp 1 A) (Matrix.detp (-1) …
    ⊢ Eq (HMul.hMul B A) 1
  -/
  replace h := congr(B * $(detp_smul_adjp hAB))
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A✝ B✝ A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    h0 : Eq (HAdd.hAdd 1 (HAdd.hAdd (HMul.hMul (Matrix.detp 1 A) (Matrix.detp (-1) …
    h : Eq (HMul.hMul B (HAdd.hAdd A (HAdd.hAdd (HSMul.hSMul (Matrix.detp 1 A) (Ma …
    ⊢ Eq (HMul.hMul B A) 1
  -/
  simp only [mul_add, mul_smul, add_assoc] at h
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A✝ B✝ A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    h0 : Eq (HAdd.hAdd 1 (HAdd.hAdd (HMul.hMul (Matrix.detp 1 A) (Matrix.detp (-1) …
    h : Eq (HAdd.hAdd (HMul.hMul B A) (HAdd.hAdd (HSMul.hSMul (Matrix.detp 1 A) (H …
    ⊢ Eq (HMul.hMul B A) 1
  -/
  replace h := congr($h + (detp 1 A * detp (-1) B + detp (-1) A * detp 1 B) • 1)
  /-
    n : Type u_1
    R : Type u_3
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A✝ B✝ A B : Matrix n n R
    hAB : Eq (HMul.hMul A B) 1
    h0 : Eq (HAdd.hAdd 1 (HAdd.hAdd (HMul.hMul (Matrix.detp 1 A) (Matrix.detp (-1) …
    h : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul B A) (HAdd.hAdd (HSMul.hSMul (Matrix.d …
    ⊢ Eq (HMul.hMul B A) 1
  -/
  simp_rw [add_smul, ← smul_smul] at h
  rwa [add_assoc, add_add_add_comm, ← smul_add, ← smul_add,
    add_add_add_comm, ← smul_add, ← smul_add, smul_add, smul_add,
    mul_adjp_add_detp, smul_add, ← mul_adjp_add_detp, smul_add, ← smul_add, ← smul_add,
    add_add_add_comm, smul_smul, smul_smul, ← add_smul, ← h0,
    add_smul, one_smul, ← add_assoc _ 1, add_comm _ 1, add_assoc,
    smul_add, smul_add, add_add_add_comm, smul_smul, smul_smul, ← add_smul,
    ((isAddUnit_detp_smul_mul_adjp hAB).add
      ((isAddUnit_detp_mul_detp hAB).smul_right _)).add_left_inj] at h


/-- We can construct an instance of invertible A if A has a left inverse. -/
def invertibleOfLeftInverse (h : B * A = 1) : Invertible A :=
  ⟨B, h, mul_eq_one_comm.mp h⟩


/-- We can construct an instance of invertible A if A has a right inverse. -/
def invertibleOfRightInverse (h : A * B = 1) : Invertible A :=
  ⟨B, mul_eq_one_comm.mp h, h⟩


theorem isUnit_of_left_inverse (h : B * A = 1) : IsUnit A :=
  ⟨⟨A, B, mul_eq_one_comm.mp h, h⟩, rfl⟩


theorem exists_left_inverse_iff_isUnit : (∃ B, B * A = 1) ↔ IsUnit A :=
  ⟨fun ⟨_, h⟩ ↦ isUnit_of_left_inverse h, fun h ↦ have := h.invertible; ⟨⅟A, invOf_mul_self' A⟩⟩


theorem isUnit_of_right_inverse (h : A * B = 1) : IsUnit A :=
  ⟨⟨A, B, h, mul_eq_one_comm.mp h⟩, rfl⟩


theorem exists_right_inverse_iff_isUnit : (∃ B, A * B = 1) ↔ IsUnit A :=
  ⟨fun ⟨_, h⟩ ↦ isUnit_of_right_inverse h, fun h ↦ have := h.invertible; ⟨⅟A, mul_invOf_self' A⟩⟩


/-- A version of `mul_eq_one_comm` that works for square matrices with rectangular types. -/
theorem mul_eq_one_comm_of_equiv {A : Matrix m n R} {B : Matrix n m R} (e : m ≃ n) :
    A * B = 1 ↔ B * A = 1 := by
  /-
    n : Type u_1
    m : Type u_2
    R : Type u_3
    inst✝⁴ : Fintype m
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : DecidableEq n
    inst✝ : CommSemiring R
    A : Matrix m n R
    B : Matrix n m R
    e : Equiv m n
    ⊢ Iff (Eq (HMul.hMul A B) 1) (Eq (HMul.hMul B A) 1)
  -/
  refine (reindex e e).injective.eq_iff.symm.trans ?_
  rw [reindex_apply, reindex_apply, submatrix_one_equiv, ← submatrix_mul_equiv _ _ _ (.refl _),
    mul_eq_one_comm, submatrix_mul_equiv, coe_refl, submatrix_id_id]


