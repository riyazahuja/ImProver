variable (α) in
/-- In a multiplicative group action, the points fixed by `g` are also fixed by `g⁻¹` -/
@[to_additive (attr := simp)
  "In an additive group action, the points fixed by `g` are also fixed by `g⁻¹`"]
theorem fixedBy_inv (g : G) : fixedBy α g⁻¹ = fixedBy α g := by
  /-
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    g : G
    ⊢ Eq (MulAction.fixedBy α (Inv.inv g)) (MulAction.fixedBy α g)
  -/
  ext
  /-
    case h
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    g : G
    x✝ : α
    ⊢ Iff (Membership.mem (MulAction.fixedBy α (Inv.inv g)) x✝) (Membership.mem (M …
  -/
  rw [mem_fixedBy, mem_fixedBy, inv_smul_eq_iff, eq_comm]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem smul_mem_fixedBy_iff_mem_fixedBy {a : α} {g : G} :
    g • a ∈ fixedBy α g ↔ a ∈ fixedBy α g := by
  /-
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : α
    g : G
    ⊢ Iff (Membership.mem (MulAction.fixedBy α g) (HSMul.hSMul g a)) (Membership.m …
  -/
  rw [mem_fixedBy, smul_left_cancel_iff]
  /-
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : α
    g : G
    ⊢ Iff (Eq (HSMul.hSMul g a) a) (Membership.mem (MulAction.fixedBy α g) a)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem smul_inv_mem_fixedBy_iff_mem_fixedBy {a : α} {g : G} :
    g⁻¹ • a ∈ fixedBy α g ↔ a ∈ fixedBy α g := by
  /-
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : α
    g : G
    ⊢ Iff (Membership.mem (MulAction.fixedBy α g) (HSMul.hSMul (Inv.inv g) a)) (Me …
  -/
  rw [← fixedBy_inv, smul_mem_fixedBy_iff_mem_fixedBy, fixedBy_inv]
  /-
    🎉 no goals
  -/


@[to_additive minimalPeriod_eq_one_iff_fixedBy]
theorem minimalPeriod_eq_one_iff_fixedBy {a : α} {g : G} :
    Function.minimalPeriod (fun x => g • x) a = 1 ↔ a ∈ fixedBy α g :=
  Function.minimalPeriod_eq_one_iff_isFixedPt


variable (α) in
@[to_additive]
theorem fixedBy_subset_fixedBy_zpow (g : G) (j : ℤ) :
    fixedBy α g ⊆ fixedBy α (g ^ j) := by
  /-
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    g : G
    j : Int
    ⊢ HasSubset.Subset (MulAction.fixedBy α g) (MulAction.fixedBy α (HPow.hPow g j))
  -/
  intro a a_in_fixedBy
  rw [mem_fixedBy, zpow_smul_eq_iff_minimalPeriod_dvd,
    minimalPeriod_eq_one_iff_fixedBy.mpr a_in_fixedBy, Nat.cast_one]
  /-
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    g : G
    j : Int
    a : α
    a_in_fixedBy : Membership.mem (MulAction.fixedBy α g) a
    ⊢ Dvd.dvd 1 j
  -/
  exact one_dvd j
  /-
    🎉 no goals
  -/


variable (M α) in
@[to_additive (attr := simp)]
theorem fixedBy_one_eq_univ : fixedBy α (1 : M) = Set.univ :=
  Set.eq_univ_iff_forall.mpr <| one_smul M


variable (α) in
@[to_additive]
theorem fixedBy_mul (m₁ m₂ : M) : fixedBy α m₁ ∩ fixedBy α m₂ ⊆ fixedBy α (m₁ * m₂) := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝¹ : Monoid M
    inst✝ : MulAction M α
    m₁ m₂ : M
    ⊢ HasSubset.Subset (Inter.inter (MulAction.fixedBy α m₁) (MulAction.fixedBy α  …
  -/
  intro a ⟨h₁, h₂⟩
  /-
    α : Type u_1
    M : Type u_3
    inst✝¹ : Monoid M
    inst✝ : MulAction M α
    m₁ m₂ : M
    a : α
    h₁ : Membership.mem (MulAction.fixedBy α m₁) a
    h₂ : Membership.mem (MulAction.fixedBy α m₂) a
    ⊢ Membership.mem (MulAction.fixedBy α (HMul.hMul m₁ m₂)) a
  -/
  rw [mem_fixedBy, mul_smul, h₂, h₁]
  /-
    🎉 no goals
  -/


variable (α) in
@[to_additive]
theorem smul_fixedBy (g h : G) :
    h • fixedBy α g = fixedBy α (h * g * h⁻¹) := by
  /-
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    g h : G
    ⊢ Eq (HSMul.hSMul h (MulAction.fixedBy α g)) (MulAction.fixedBy α (HMul.hMul ( …
  -/
  ext a
  /-
    case h
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    g h : G
    a : α
    ⊢ Iff (Membership.mem (HSMul.hSMul h (MulAction.fixedBy α g)) a) (Membership.m …
  -/
  simp_rw [Set.mem_smul_set_iff_inv_smul_mem, mem_fixedBy, mul_smul, smul_eq_iff_eq_inv_smul h]
  /-
    🎉 no goals
  -/


/--
If a set `s : Set α` is in `fixedBy (Set α) g`, then all points of `s` will stay in `s` after being
moved by `g`.
-/
@[to_additive "If a set `s : Set α` is in `fixedBy (Set α) g`, then all points of `s` will stay in
`s` after being moved by `g`."]
theorem set_mem_fixedBy_iff (s : Set α) (g : G) :
    s ∈ fixedBy (Set α) g ↔ ∀ x, g • x ∈ s ↔ x ∈ s := by
  /-
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    g : G
    ⊢ Iff (Membership.mem (MulAction.fixedBy (Set α) g) s) (∀ (x : α), Iff (Member …
  -/
  simp_rw [mem_fixedBy, ← eq_inv_smul_iff, Set.ext_iff, Set.mem_inv_smul_set_iff, Iff.comm]
  /-
    🎉 no goals
  -/


theorem smul_mem_of_set_mem_fixedBy {s : Set α} {g : G} (s_in_fixedBy : s ∈ fixedBy (Set α) g)
    {x : α} : g • x ∈ s ↔ x ∈ s := (set_mem_fixedBy_iff s g).mp s_in_fixedBy x


/--
If `s ⊆ fixedBy α g`, then `g • s = s`, which means that `s ∈ fixedBy (Set α) g`.

Note that the reverse implication is in general not true, as `s ∈ fixedBy (Set α) g` is a
weaker statement (it allows for points `x ∈ s` for which `g • x ≠ x` and `g • x ∈ s`).
-/
@[to_additive "If `s ⊆ fixedBy α g`, then `g +ᵥ s = s`, which means that `s ∈ fixedBy (Set α) g`.

Note that the reverse implication is in general not true, as `s ∈ fixedBy (Set α) g` is a
weaker statement (it allows for points `x ∈ s` for which `g +ᵥ x ≠ x` and `g +ᵥ x ∈ s`)."]
theorem set_mem_fixedBy_of_subset_fixedBy {s : Set α} {g : G} (s_ss_fixedBy : s ⊆ fixedBy α g) :
    s ∈ fixedBy (Set α) g := by
  /-
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    g : G
    s_ss_fixedBy : HasSubset.Subset s (MulAction.fixedBy α g)
    ⊢ Membership.mem (MulAction.fixedBy (Set α) g) s
  -/
  rw [← fixedBy_inv]
  /-
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    g : G
    s_ss_fixedBy : HasSubset.Subset s (MulAction.fixedBy α g)
    ⊢ Membership.mem (MulAction.fixedBy (Set α) (Inv.inv g)) s
  -/
  ext x
  /-
    case h
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    g : G
    s_ss_fixedBy : HasSubset.Subset s (MulAction.fixedBy α g)
    x : α
    ⊢ Iff (Membership.mem (HSMul.hSMul (Inv.inv g) s) x) (Membership.mem s x)
  -/
  rw [Set.mem_inv_smul_set_iff]
  /-
    case h
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    g : G
    s_ss_fixedBy : HasSubset.Subset s (MulAction.fixedBy α g)
    x : α
    ⊢ Iff (Membership.mem s (HSMul.hSMul g x)) (Membership.mem s x)
  -/
  refine ⟨fun gxs => ?xs, fun xs => (s_ss_fixedBy xs).symm ▸ xs⟩
  /-
    case xs
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    g : G
    s_ss_fixedBy : HasSubset.Subset s (MulAction.fixedBy α g)
    x : α
    gxs : Membership.mem s (HSMul.hSMul g x)
    ⊢ Membership.mem s x
  -/
  rw [← fixedBy_inv] at s_ss_fixedBy
  /-
    case xs
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    g : G
    s_ss_fixedBy : HasSubset.Subset s (MulAction.fixedBy α (Inv.inv g))
    x : α
    gxs : Membership.mem s (HSMul.hSMul g x)
    ⊢ Membership.mem s x
  -/
  rwa [← s_ss_fixedBy gxs, inv_smul_smul] at gxs
  /-
    🎉 no goals
  -/


theorem smul_subset_of_set_mem_fixedBy {s t : Set α} {g : G} (t_ss_s : t ⊆ s)
    (s_in_fixedBy : s ∈ fixedBy (Set α) g) : g • t ⊆ s :=
  (Set.set_smul_subset_set_smul_iff.mpr t_ss_s).trans s_in_fixedBy.subset


/-- If `(fixedBy α g)ᶜ ⊆ s`, then `g` cannot move a point of `s` outside of `s`. -/
@[to_additive "If `(fixedBy α g)ᶜ ⊆ s`, then `g` cannot move a point of `s` outside of `s`."]
theorem set_mem_fixedBy_of_movedBy_subset {s : Set α} {g : G} (s_subset : (fixedBy α g)ᶜ ⊆ s) :
    s ∈ fixedBy (Set α) g := by
  /-
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    g : G
    s_subset : HasSubset.Subset (HasCompl.compl (MulAction.fixedBy α g)) s
    ⊢ Membership.mem (MulAction.fixedBy (Set α) g) s
  -/
  rw [← fixedBy_inv]
  /-
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    g : G
    s_subset : HasSubset.Subset (HasCompl.compl (MulAction.fixedBy α g)) s
    ⊢ Membership.mem (MulAction.fixedBy (Set α) (Inv.inv g)) s
  -/
  ext a
  /-
    case h
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    g : G
    s_subset : HasSubset.Subset (HasCompl.compl (MulAction.fixedBy α g)) s
    a : α
    ⊢ Iff (Membership.mem (HSMul.hSMul (Inv.inv g) s) a) (Membership.mem s a)
  -/
  rw [Set.mem_inv_smul_set_iff]
  /-
    case h
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    g : G
    s_subset : HasSubset.Subset (HasCompl.compl (MulAction.fixedBy α g)) s
    a : α
    ⊢ Iff (Membership.mem s (HSMul.hSMul g a)) (Membership.mem s a)
  -/
  by_cases a ∈ fixedBy α g
  case pos a_fixed =>
    rw [a_fixed]
  case neg a_moved =>
    constructor <;> (intro; apply s_subset)
    · exact a_moved
    · rwa [Set.mem_compl_iff, smul_mem_fixedBy_iff_mem_fixedBy]


/--
If `g` and `h` commute, then `g` fixes `h • x` iff `g` fixes `x`.
This is equivalent to say that the set `fixedBy α g` is fixed by `h`.
-/
@[to_additive "If `g` and `h` commute, then `g` fixes `h +ᵥ x` iff `g` fixes `x`.
This is equivalent to say that the set `fixedBy α g` is fixed by `h`.
"]
theorem fixedBy_mem_fixedBy_of_commute {g h : G} (comm : Commute g h) :
    (fixedBy α g) ∈ fixedBy (Set α) h := by
  /-
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    g h : G
    comm : Commute g h
    ⊢ Membership.mem (MulAction.fixedBy (Set α) h) (MulAction.fixedBy α g)
  -/
  ext x
  rw [Set.mem_smul_set_iff_inv_smul_mem, mem_fixedBy, ← mul_smul, comm.inv_right, mul_smul,
    smul_left_cancel_iff, mem_fixedBy]


/--
If `g` and `h` commute, then `g` fixes `(h ^ j) • x` iff `g` fixes `x`.
-/
@[to_additive "If `g` and `h` commute, then `g` fixes `(j • h) +ᵥ x` iff `g` fixes `x`."]
theorem smul_zpow_fixedBy_eq_of_commute {g h : G} (comm : Commute g h) (j : ℤ) :
    h ^ j • fixedBy α g = fixedBy α g :=
  fixedBy_subset_fixedBy_zpow (Set α) h j (fixedBy_mem_fixedBy_of_commute comm)


/--
If `g` and `h` commute, then `g` moves `h • x` iff `g` moves `x`.
This is equivalent to say that the set `(fixedBy α g)ᶜ` is fixed by `h`.
-/
@[to_additive "If `g` and `h` commute, then `g` moves `h +ᵥ x` iff `g` moves `x`.
This is equivalent to say that the set `(fixedBy α g)ᶜ` is fixed by `h`."]
theorem movedBy_mem_fixedBy_of_commute {g h : G} (comm : Commute g h) :
    (fixedBy α g)ᶜ ∈ fixedBy (Set α) h := by
  /-
    α : Type u_1
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : MulAction G α
    g h : G
    comm : Commute g h
    ⊢ Membership.mem (MulAction.fixedBy (Set α) h) (HasCompl.compl (MulAction.fixe …
  -/
  rw [mem_fixedBy, Set.smul_set_compl, fixedBy_mem_fixedBy_of_commute comm]
  /-
    🎉 no goals
  -/


/--
If `g` and `h` commute, then `g` moves `h ^ j • x` iff `g` moves `x`.
-/
@[to_additive "If `g` and `h` commute, then `g` moves `(j • h) +ᵥ x` iff `g` moves `x`."]
theorem smul_zpow_movedBy_eq_of_commute {g h : G} (comm : Commute g h) (j : ℤ) :
    h ^ j • (fixedBy α g)ᶜ = (fixedBy α g)ᶜ :=
  fixedBy_subset_fixedBy_zpow (Set α) h j (movedBy_mem_fixedBy_of_commute comm)


/-- If the multiplicative action of `M` on `α` is faithful,
then `fixedBy α m = Set.univ` implies that `m = 1`. -/
@[to_additive "If the additive action of `M` on `α` is faithful,
then `fixedBy α m = Set.univ` implies that `m = 1`."]
theorem fixedBy_eq_univ_iff_eq_one {m : M} : fixedBy α m = Set.univ ↔ m = 1 := by
  /-
    α : Type u_1
    M : Type u_3
    inst✝² : Monoid M
    inst✝¹ : MulAction M α
    inst✝ : FaithfulSMul M α
    m : M
    ⊢ Iff (Eq (MulAction.fixedBy α m) Set.univ) (Eq m 1)
  -/
  rw [← (smul_left_injective' (M := M) (α := α)).eq_iff, Set.eq_univ_iff_forall]
  /-
    α : Type u_1
    M : Type u_3
    inst✝² : Monoid M
    inst✝¹ : MulAction M α
    inst✝ : FaithfulSMul M α
    m : M
    ⊢ Iff (∀ (x : α), Membership.mem (MulAction.fixedBy α m) x) (Eq (fun x2 => HSM …
  -/
  simp_rw [funext_iff, one_smul, mem_fixedBy]
  /-
    🎉 no goals
  -/


/--
If the image of the `(fixedBy α g)ᶜ` set by the pointwise action of `h: G`
is disjoint from `(fixedBy α g)ᶜ`, then `g` and `h` cannot commute.
-/
@[to_additive "If the image of the `(fixedBy α g)ᶜ` set by the pointwise action of `h: G`
is disjoint from `(fixedBy α g)ᶜ`, then `g` and `h` cannot commute."]
theorem not_commute_of_disjoint_movedBy_preimage {g h : G} (ne_one : g ≠ 1)
    (disjoint : Disjoint (fixedBy α g)ᶜ (h • (fixedBy α g)ᶜ)) : ¬Commute g h := by
  /-
    α : Type u_1
    G : Type u_2
    inst✝² : Group G
    inst✝¹ : MulAction G α
    inst✝ : FaithfulSMul G α
    g h : G
    ne_one : Ne g 1
    disjoint : Disjoint (HasCompl.compl (MulAction.fixedBy α g)) (HSMul.hSMul h (H …
    ⊢ Not (Commute g h)
  -/
  contrapose! ne_one with comm
  rwa [movedBy_mem_fixedBy_of_commute comm, disjoint_self, Set.bot_eq_empty, ← Set.compl_univ,
    compl_inj_iff, fixedBy_eq_univ_iff_eq_one] at disjoint


