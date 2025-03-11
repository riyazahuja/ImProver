theorem List.smul_sum {r : α} {l : List β} : r • l.sum = (l.map (r • ·)).sum :=
  map_list_sum (DistribSMul.toAddMonoidHom β r) l


theorem List.smul_prod' {r : α} {l : List β} : r • l.prod = (l.map (r • ·)).prod :=
  map_list_prod (MulDistribMulAction.toMonoidHom β r) l


theorem Multiset.smul_sum {r : α} {s : Multiset β} : r • s.sum = (s.map (r • ·)).sum :=
  (DistribSMul.toAddMonoidHom β r).map_multiset_sum s


theorem Finset.smul_sum {r : α} {f : γ → β} {s : Finset γ} :
    (r • ∑ x ∈ s, f x) = ∑ x ∈ s, r • f x :=
  map_sum (DistribSMul.toAddMonoidHom β r) f s


theorem Multiset.smul_prod' {r : α} {s : Multiset β} : r • s.prod = (s.map (r • ·)).prod :=
  (MulDistribMulAction.toMonoidHom β r).map_multiset_prod s


theorem Finset.smul_prod' {r : α} {f : γ → β} {s : Finset γ} :
    (r • ∏ x ∈ s, f x) = ∏ x ∈ s, r • f x :=
  map_prod (MulDistribMulAction.toMonoidHom β r) f s


theorem smul_finprod' {ι : Sort*} [Finite ι] {f : ι → β} (r : α) :
    r • ∏ᶠ x : ι, f x = ∏ᶠ x : ι, r • (f x) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Monoid α
    inst✝² : CommMonoid β
    inst✝¹ : MulDistribMulAction α β
    ι : Sort u_4
    inst✝ : Finite ι
    f : ι → β
    r : α
    ⊢ Eq (HSMul.hSMul r (finprod fun x => f x)) (finprod fun x => HSMul.hSMul r (f …
  -/
  cases nonempty_fintype (PLift ι)
  simp only [finprod_eq_prod_plift_of_mulSupport_subset (s := Finset.univ) (by simp),
    finprod_eq_prod_of_fintype, Finset.smul_prod']


theorem Finset.smul_prod_perm [Fintype G] (b : β) (g : G) :
    (g • ∏ h : G, h • b) = ∏ h : G, h • b := by
  /-
    β : Type u_2
    inst✝³ : CommMonoid β
    G : Type u_4
    inst✝² : Group G
    inst✝¹ : MulDistribMulAction G β
    inst✝ : Fintype G
    b : β
    g : G
    ⊢ Eq (HSMul.hSMul g (Finset.univ.prod fun h => HSMul.hSMul h b)) (Finset.univ. …
  -/
  simp only [smul_prod', smul_smul]
  /-
    β : Type u_2
    inst✝³ : CommMonoid β
    G : Type u_4
    inst✝² : Group G
    inst✝¹ : MulDistribMulAction G β
    inst✝ : Fintype G
    b : β
    g : G
    ⊢ Eq (Finset.univ.prod fun x => HSMul.hSMul (HMul.hMul g x) b) (Finset.univ.pr …
  -/
  exact Finset.prod_bijective (g * ·) (Group.mulLeft_bijective g) (by simp) (fun _ _ ↦ rfl)
  /-
    🎉 no goals
  -/


theorem smul_finprod_perm [Finite G] (b : β) (g : G) :
    (g • ∏ᶠ h : G, h • b) = ∏ᶠ h : G, h • b := by
  /-
    β : Type u_2
    inst✝³ : CommMonoid β
    G : Type u_4
    inst✝² : Group G
    inst✝¹ : MulDistribMulAction G β
    inst✝ : Finite G
    b : β
    g : G
    ⊢ Eq (HSMul.hSMul g (finprod fun h => HSMul.hSMul h b)) (finprod fun h => HSMu …
  -/
  cases nonempty_fintype G
  /-
    case intro
    β : Type u_2
    inst✝³ : CommMonoid β
    G : Type u_4
    inst✝² : Group G
    inst✝¹ : MulDistribMulAction G β
    inst✝ : Finite G
    b : β
    g : G
    val✝ : Fintype G
    ⊢ Eq (HSMul.hSMul g (finprod fun h => HSMul.hSMul h b)) (finprod fun h => HSMu …
  -/
  simp only [finprod_eq_prod_of_fintype, Finset.smul_prod_perm]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem smul_prod [Monoid α] [Monoid β] [MulAction α β] [IsScalarTower α β β] [SMulCommClass α β β]
    (l : List β) (m : α) :
    m ^ l.length • l.prod = (l.map (m • ·)).prod := by
  induction l with
  | nil => simp
  | cons head tail ih => simp [← ih, smul_mul_smul_comm, pow_succ']


@[to_additive]
theorem smul_prod [Monoid α] [CommMonoid β] [MulAction α β] [IsScalarTower α β β]
    [SMulCommClass α β β] (s : Multiset β) (b : α) :
    b ^ card s • s.prod = (s.map (b • ·)).prod :=
                            /-
                              α : Type u_1
                              β : Type u_2
                              inst✝⁴ : Monoid α
                              inst✝³ : CommMonoid β
                              inst✝² : MulAction α β
                              inst✝¹ : IsScalarTower α β β
                              inst✝ : SMulCommClass α β β
                              s : Multiset β
                              b : α
                              ⊢ ∀ (a : List β), Eq (HSMul.hSMul (HPow.hPow b (Multiset.card (Quot.mk (⇑(List …
                            -/
  Quot.induction_on s <| by simp [List.smul_prod]
                            /-
                              🎉 no goals
                            -/


theorem smul_prod
    [CommMonoid β] [Monoid α] [MulAction α β] [IsScalarTower α β β] [SMulCommClass α β β]
    (s : Finset β) (b : α) (f : β → β) :
    b ^ s.card • ∏ x in s, f x = ∏ x in s, b • f x := by
  have : Multiset.map (fun (x : β) ↦ b • f x) s.val =
      Multiset.map (fun x ↦ b • x) (Multiset.map f s.val) := by
    simp only [Multiset.map_map, Function.comp_apply]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : CommMonoid β
    inst✝³ : Monoid α
    inst✝² : MulAction α β
    inst✝¹ : IsScalarTower α β β
    inst✝ : SMulCommClass α β β
    s : Finset β
    b : α
    f : β → β
    this : Eq (Multiset.map (fun x => HSMul.hSMul b (f x)) s.val) (Multiset.map (f …
    ⊢ Eq (HSMul.hSMul (HPow.hPow b s.card) (s.prod fun x => f x)) (s.prod fun x => …
  -/
  simp_rw [prod_eq_multiset_prod, card_def, this, ← Multiset.smul_prod _ b, Multiset.card_map]
  /-
    🎉 no goals
  -/


theorem prod_smul
    [CommMonoid β] [CommMonoid α] [MulAction α β] [IsScalarTower α β β] [SMulCommClass α β β]
    (s : Finset β) (b : β → α) (f : β → β) :
    ∏ i in s, b i • f i = (∏ i in s, b i) • ∏ i in s, f i := by
  induction s using Finset.cons_induction_on with
  | h₁ =>  simp
  | h₂ hj ih => rw [prod_cons, ih, smul_mul_smul_comm, ← prod_cons hj, ← prod_cons hj]


