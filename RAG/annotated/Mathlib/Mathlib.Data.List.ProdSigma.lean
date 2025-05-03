@[simp]
theorem nil_product (l : List β) : (@nil α) ×ˢ l = [] :=
  rfl


@[simp]
theorem product_cons (a : α) (l₁ : List α) (l₂ : List β) :
    (a :: l₁) ×ˢ l₂ = map (fun b => (a, b)) l₂ ++ (l₁ ×ˢ l₂) :=
  rfl


@[simp]
theorem product_nil : ∀ l : List α, l ×ˢ (@nil β) = []
  | [] => rfl
                 /-
                   α : Type u_1
                   β : Type u_2
                   head✝ : α
                   l : List α
                   ⊢ Eq (SProd.sprod (List.cons head✝ l) List.nil) List.nil
                 -/
  | _ :: l => by simp [product_cons, product_nil l]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem mem_product {l₁ : List α} {l₂ : List β} {a : α} {b : β} :
    (a, b) ∈ l₁ ×ˢ l₂ ↔ a ∈ l₁ ∧ b ∈ l₂ := by
  simp_all [SProd.sprod, product, mem_flatMap, mem_map, Prod.ext_iff, exists_prop, and_left_comm,
    exists_and_left, exists_eq_left, exists_eq_right]


theorem length_product (l₁ : List α) (l₂ : List β) :
    length (l₁ ×ˢ l₂) = length l₁ * length l₂ := by
  /-
    α : Type u_1
    β : Type u_2
    l₁ : List α
    l₂ : List β
    ⊢ Eq (SProd.sprod l₁ l₂).length (HMul.hMul l₁.length l₂.length)
  -/
  induction' l₁ with x l₁ IH
    /-
      case nil
      α : Type u_1
      β : Type u_2
      l₂ : List β
      ⊢ Eq (SProd.sprod List.nil l₂).length (HMul.hMul List.nil.length l₂.length)
    -/
  · exact (Nat.zero_mul _).symm
    /-
      🎉 no goals
    -/
  · simp only [length, product_cons, length_append, IH, Nat.add_mul, Nat.one_mul, length_map,
      Nat.add_comm]


@[simp]
theorem nil_sigma (l : ∀ a, List (σ a)) : (@nil α).sigma l = [] :=
  rfl


@[simp]
theorem sigma_cons (a : α) (l₁ : List α) (l₂ : ∀ a, List (σ a)) :
    (a :: l₁).sigma l₂ = map (Sigma.mk a) (l₂ a) ++ l₁.sigma l₂ :=
  rfl


@[simp]
theorem sigma_nil : ∀ l : List α, (l.sigma fun a => @nil (σ a)) = []
  | [] => rfl
                 /-
                   α : Type u_1
                   σ : α → Type u_3
                   head✝ : α
                   l : List α
                   ⊢ Eq ((List.cons head✝ l).sigma fun a => List.nil) List.nil
                 -/
  | _ :: l => by simp [sigma_cons, sigma_nil l]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem mem_sigma {l₁ : List α} {l₂ : ∀ a, List (σ a)} {a : α} {b : σ a} :
    Sigma.mk a b ∈ l₁.sigma l₂ ↔ a ∈ l₁ ∧ b ∈ l₂ a := by
  simp [List.sigma, mem_flatMap, mem_map, exists_prop, exists_and_left, and_left_comm,
    exists_eq_left, heq_iff_eq, exists_eq_right]


set_option linter.deprecated false in
/-- See `List.length_sigma` for the corresponding statement using `List.sum`. -/
@[deprecated "Use `List.length_sigma`." (since := "2024-10-17")]
theorem length_sigma' (l₁ : List α) (l₂ : ∀ a, List (σ a)) :
    length (l₁.sigma l₂) = Nat.sum (l₁.map fun a ↦ length (l₂ a)) := by
  /-
    α : Type u_1
    σ : α → Type u_3
    l₁ : List α
    l₂ : (a : α) → List (σ a)
    ⊢ Eq (l₁.sigma l₂).length (Nat.sum (List.map (fun a => (l₂ a).length) l₁))
  -/
  induction' l₁ with x l₁ IH
    /-
      case nil
      α : Type u_1
      σ : α → Type u_3
      l₂ : (a : α) → List (σ a)
      ⊢ Eq (List.nil.sigma l₂).length (Nat.sum (List.map (fun a => (l₂ a).length) Li …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      σ : α → Type u_3
      l₂ : (a : α) → List (σ a)
      x : α
      l₁ : List α
      IH : Eq (l₁.sigma l₂).length (Nat.sum (List.map (fun a => (l₂ a).length) l₁))
      ⊢ Eq ((List.cons x l₁).sigma l₂).length (Nat.sum (List.map (fun a => (l₂ a).le …
    -/
  · simp only [map, sigma_cons, length_append, length_map, IH, Nat.sum_cons]
    /-
      🎉 no goals
    -/


