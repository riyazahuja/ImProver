@[to_additive (attr := simp, norm_cast)]
theorem inv_coe_set [InvolutiveInv G] [SetLike S G] [InvMemClass S G] {H : S} : (H : Set G)⁻¹ = H :=
  Set.ext fun _ => inv_mem_iff


@[to_additive (attr := simp)]
lemma smul_coe_set [Group G] [SetLike S G] [SubgroupClass S G] {s : S} {a : G} (ha : a ∈ s) :
    a • (s : Set G) = s := by
  /-
    G : Type u_2
    S : Type u_4
    inst✝² : Group G
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    s : S
    a : G
    ha : Membership.mem s a
    ⊢ Eq (HSMul.hSMul a ↑s) ↑s
  -/
  ext; simp [Set.mem_smul_set_iff_inv_smul_mem, mul_mem_cancel_left, ha]
       /-
         🎉 no goals
       -/


@[norm_cast, to_additive]
lemma coe_set_eq_one [Group G] {s : Subgroup G} : (s : Set G) = 1 ↔ s = ⊥ :=
                              /-
                                G : Type u_2
                                inst✝ : Group G
                                s : Subgroup G
                                ⊢ Iff (Eq ↑s ↑Bot.bot) (Eq (↑s) 1)
                              -/
  (SetLike.ext'_iff.trans (by rfl)).symm
                              /-
                                🎉 no goals
                              -/


@[to_additive (attr := simp)]
lemma op_smul_coe_set [Group G] [SetLike S G] [SubgroupClass S G] {s : S} {a : G} (ha : a ∈ s) :
    MulOpposite.op a • (s : Set G) = s := by
  /-
    G : Type u_2
    S : Type u_4
    inst✝² : Group G
    inst✝¹ : SetLike S G
    inst✝ : SubgroupClass S G
    s : S
    a : G
    ha : Membership.mem s a
    ⊢ Eq (HSMul.hSMul (MulOpposite.op a) ↑s) ↑s
  -/
  ext; simp [Set.mem_smul_set_iff_inv_smul_mem, mul_mem_cancel_right, ha]
       /-
         🎉 no goals
       -/


@[to_additive (attr := simp, norm_cast)]
lemma coe_div_coe [SetLike S G] [DivisionMonoid G] [SubgroupClass S G] (H : S) :
                              /-
                                G : Type u_2
                                S : Type u_4
                                inst✝² : SetLike S G
                                inst✝¹ : DivisionMonoid G
                                inst✝ : SubgroupClass S G
                                H : S
                                ⊢ Eq (HDiv.hDiv ↑H ↑H) ↑H
                              -/
    H / H = (H : Set G) := by simp [div_eq_mul_inv]
                              /-
                                🎉 no goals
                              -/


@[to_additive (attr := simp)]
lemma mul_subgroupClosure (hs : s.Nonempty) : s * closure s = closure s := by
  /-
    G : Type u_2
    inst✝ : Group G
    s : Set G
    hs : s.Nonempty
    ⊢ Eq (HMul.hMul s ↑(Subgroup.closure s)) ↑(Subgroup.closure s)
  -/
  rw [← smul_eq_mul, ← Set.iUnion_smul_set]
  have h a (ha : a ∈ s) : a • (closure s : Set G) = closure s :=
    smul_coe_set <| subset_closure ha
  /-
    G : Type u_2
    inst✝ : Group G
    s : Set G
    hs : s.Nonempty
    h : ∀ (a : G), Membership.mem s a → Eq (HSMul.hSMul a ↑(Subgroup.closure s)) ↑ …
    ⊢ Eq (Set.iUnion fun a => Set.iUnion fun h => HSMul.hSMul a ↑(Subgroup.closure …
  -/
  simp +contextual [h, hs]
  /-
    🎉 no goals
  -/


open scoped RightActions in
@[to_additive (attr := simp)]
lemma subgroupClosure_mul (hs : s.Nonempty) : closure s * s = closure s := by
  /-
    G : Type u_2
    inst✝ : Group G
    s : Set G
    hs : s.Nonempty
    ⊢ Eq (HMul.hMul (↑(Subgroup.closure s)) s) ↑(Subgroup.closure s)
  -/
  rw [← Set.iUnion_op_smul_set]
  have h a (ha : a ∈ s) :  (closure s : Set G) <• a = closure s :=
    op_smul_coe_set <| subset_closure ha
  /-
    G : Type u_2
    inst✝ : Group G
    s : Set G
    hs : s.Nonempty
    h : ∀ (a : G), Membership.mem s a → Eq (HSMul.hSMul (MulOpposite.op a) ↑(Subgr …
    ⊢ Eq (Set.iUnion fun a => Set.iUnion fun h => HSMul.hSMul (MulOpposite.op a) ↑ …
  -/
  simp +contextual [h, hs]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma pow_mul_subgroupClosure (hs : s.Nonempty) : ∀ n, s ^ n * closure s = closure s
            /-
              G : Type u_2
              inst✝ : Group G
              s : Set G
              hs : s.Nonempty
              ⊢ Eq (HMul.hMul (HPow.hPow s 0) ↑(Subgroup.closure s)) ↑(Subgroup.closure s)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
                /-
                  G : Type u_2
                  inst✝ : Group G
                  s : Set G
                  hs : s.Nonempty
                  n : Nat
                  ⊢ Eq (HMul.hMul (HPow.hPow s (HAdd.hAdd n 1)) ↑(Subgroup.closure s)) ↑(Subgrou …
                -/
  | n + 1 => by rw [pow_succ, mul_assoc, mul_subgroupClosure hs, pow_mul_subgroupClosure hs]
                /-
                  🎉 no goals
                -/


@[to_additive (attr := simp)]
lemma subgroupClosure_mul_pow (hs : s.Nonempty) : ∀ n, closure s * s ^ n = closure s
            /-
              G : Type u_2
              inst✝ : Group G
              s : Set G
              hs : s.Nonempty
              ⊢ Eq (HMul.hMul (↑(Subgroup.closure s)) (HPow.hPow s 0)) ↑(Subgroup.closure s)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
                /-
                  G : Type u_2
                  inst✝ : Group G
                  s : Set G
                  hs : s.Nonempty
                  n : Nat
                  ⊢ Eq (HMul.hMul (↑(Subgroup.closure s)) (HPow.hPow s (HAdd.hAdd n 1))) ↑(Subgr …
                -/
  | n + 1 => by rw [pow_succ', ← mul_assoc, subgroupClosure_mul hs, subgroupClosure_mul_pow hs]
                /-
                  🎉 no goals
                -/


@[to_additive (attr := simp)]
theorem inv_subset_closure (S : Set G) : S⁻¹ ⊆ closure S := fun s hs => by
  /-
    G : Type u_2
    inst✝ : Group G
    S : Set G
    s : G
    hs : Membership.mem (Inv.inv S) s
    ⊢ Membership.mem (↑(Subgroup.closure S)) s
  -/
  rw [SetLike.mem_coe, ← Subgroup.inv_mem_iff]
  /-
    G : Type u_2
    inst✝ : Group G
    S : Set G
    s : G
    hs : Membership.mem (Inv.inv S) s
    ⊢ Membership.mem (Subgroup.closure S) (Inv.inv s)
  -/
  exact subset_closure (mem_inv.mp hs)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem closure_toSubmonoid (S : Set G) :
    (closure S).toSubmonoid = Submonoid.closure (S ∪ S⁻¹) := by
  /-
    G : Type u_2
    inst✝ : Group G
    S : Set G
    ⊢ Eq (Subgroup.closure S).toSubmonoid (Submonoid.closure (Union.union S (Inv.i …
  -/
  refine le_antisymm (fun x hx => ?_) (Submonoid.closure_le.2 ?_)
  · refine
      closure_induction
        (fun x hx => Submonoid.closure_mono subset_union_left (Submonoid.subset_closure hx))
        (Submonoid.one_mem _) (fun x y _ _ hx hy => Submonoid.mul_mem _ hx hy) (fun x _ hx => ?_) hx
    /-
      case refine_1
      G : Type u_2
      inst✝ : Group G
      S : Set G
      x✝¹ : G
      hx✝ : Membership.mem (Subgroup.closure S).toSubmonoid x✝¹
      x : G
      x✝ : Membership.mem (Subgroup.closure S) x
      hx : Membership.mem (Submonoid.closure (Union.union S (Inv.inv S))) x
      ⊢ Membership.mem (Submonoid.closure (Union.union S (Inv.inv S))) (Inv.inv x)
    -/
    rwa [← Submonoid.mem_closure_inv, Set.union_inv, inv_inv, Set.union_comm]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_2
      inst✝ : Group G
      S : Set G
      ⊢ HasSubset.Subset (Union.union S (Inv.inv S)) ↑(Subgroup.closure S).toSubmonoid
    -/
  · simp only [true_and, coe_toSubmonoid, union_subset_iff, subset_closure, inv_subset_closure]
    /-
      🎉 no goals
    -/


/-- For subgroups generated by a single element, see the simpler `zpow_induction_left`. -/
@[to_additive (attr := elab_as_elim)
  "For additive subgroups generated by a single element, see the simpler
  `zsmul_induction_left`."]
theorem closure_induction_left {p : (x : G) → x ∈ closure s → Prop} (one : p 1 (one_mem _))
    (mul_left : ∀ x (hx : x ∈ s), ∀ (y) hy, p y hy → p (x * y) (mul_mem (subset_closure hx) hy))
    (inv_mul_cancel : ∀ x (hx : x ∈ s), ∀ (y) hy, p y hy →
      p (x⁻¹ * y) (mul_mem (inv_mem (subset_closure hx)) hy))
    {x : G} (h : x ∈ closure s) : p x h := by
  /-
    G : Type u_2
    inst✝ : Group G
    s : Set G
    p : (x : G) → Membership.mem (Subgroup.closure s) x → Prop
    one : p 1 ⋯
    mul_left : ∀ (x : G) (hx : Membership.mem s x) (y : G) (hy : Membership.mem (S …
    inv_mul_cancel : ∀ (x : G) (hx : Membership.mem s x) (y : G) (hy : Membership. …
    x : G
    h : Membership.mem (Subgroup.closure s) x
    ⊢ p x h
  -/
  revert h
  /-
    G : Type u_2
    inst✝ : Group G
    s : Set G
    p : (x : G) → Membership.mem (Subgroup.closure s) x → Prop
    one : p 1 ⋯
    mul_left : ∀ (x : G) (hx : Membership.mem s x) (y : G) (hy : Membership.mem (S …
    inv_mul_cancel : ∀ (x : G) (hx : Membership.mem s x) (y : G) (hy : Membership. …
    x : G
    ⊢ ∀ (h : Membership.mem (Subgroup.closure s) x), p x h
  -/
  simp_rw [← mem_toSubmonoid, closure_toSubmonoid] at *
  /-
    G : Type u_2
    inst✝ : Group G
    s : Set G
    p : (x : G) → Membership.mem (Subgroup.closure s) x → Prop
    one : p 1 ⋯
    x : G
    mul_left : ∀ (x : G) (hx : Membership.mem s x) (y : G) (hy : Membership.mem (S …
    inv_mul_cancel : ∀ (x : G) (hx : Membership.mem s x) (y : G) (hy : Membership. …
    ⊢ ∀ (h : Membership.mem (Submonoid.closure (Union.union s (Inv.inv s))) x), p  …
  -/
  intro h
  induction h using Submonoid.closure_induction_left with
  | one => exact one
  | mul_left x hx y hy ih =>
    cases hx with
    | inl hx => exact mul_left _ hx _ hy ih
    | inr hx => simpa only [inv_inv] using inv_mul_cancel _ hx _ hy ih


/-- For subgroups generated by a single element, see the simpler `zpow_induction_right`. -/
@[to_additive (attr := elab_as_elim)
  "For additive subgroups generated by a single element, see the simpler
  `zsmul_induction_right`."]
theorem closure_induction_right {p : (x : G) → x ∈ closure s → Prop} (one : p 1 (one_mem _))
    (mul_right : ∀ (x) hx, ∀ y (hy : y ∈ s), p x hx → p (x * y) (mul_mem hx (subset_closure hy)))
    (mul_inv_cancel : ∀ (x) hx, ∀ y (hy : y ∈ s), p x hx →
      p (x * y⁻¹) (mul_mem hx (inv_mem (subset_closure hy))))
    {x : G} (h : x ∈ closure s) : p x h :=
  closure_induction_left (s := MulOpposite.unop ⁻¹' s)
                                     /-
                                       G : Type u_2
                                       inst✝ : Group G
                                       s : Set G
                                       p : (x : G) → Membership.mem (Subgroup.closure s) x → Prop
                                       one : p 1 ⋯
                                       mul_right : ∀ (x : G) (hx : Membership.mem (Subgroup.closure s) x) (y : G) (hy …
                                       mul_inv_cancel : ∀ (x : G) (hx : Membership.mem (Subgroup.closure s) x) (y : G …
                                       x : G
                                       h : Membership.mem (Subgroup.closure s) x
                                       m : MulOpposite G
                                       hm : Membership.mem (Subgroup.closure (Set.preimage MulOpposite.unop s)) m
                                       ⊢ Membership.mem (Subgroup.closure s) (MulOpposite.unop m)
                                     -/
    (p := fun m hm => p m.unop <| by rwa [← op_closure] at hm)
                                     /-
                                       🎉 no goals
                                     -/
    one
    (fun _x hx _y _ => mul_right _ _ _ hx)
    (fun _x hx _y _ => mul_inv_cancel _ _ _ hx)
        /-
          G : Type u_2
          inst✝ : Group G
          s : Set G
          p : (x : G) → Membership.mem (Subgroup.closure s) x → Prop
          one : p 1 ⋯
          mul_right : ∀ (x : G) (hx : Membership.mem (Subgroup.closure s) x) (y : G) (hy …
          mul_inv_cancel : ∀ (x : G) (hx : Membership.mem (Subgroup.closure s) x) (y : G …
          x : G
          h : Membership.mem (Subgroup.closure s) x
          ⊢ Membership.mem (Subgroup.closure (Set.preimage MulOpposite.unop s)) { unop'  …
        -/
    (by rwa [← op_closure])
        /-
          🎉 no goals
        -/


@[to_additive (attr := simp)]
theorem closure_inv (s : Set G) : closure s⁻¹ = closure s := by
  /-
    G : Type u_2
    inst✝ : Group G
    s : Set G
    ⊢ Eq (Subgroup.closure (Inv.inv s)) (Subgroup.closure s)
  -/
  simp only [← toSubmonoid_inj, closure_toSubmonoid, inv_inv, union_comm]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma closure_singleton_inv (x : G) : closure {x⁻¹} = closure {x} := by
  /-
    G : Type u_2
    inst✝ : Group G
    x : G
    ⊢ Eq (Subgroup.closure (Singleton.singleton (Inv.inv x))) (Subgroup.closure (S …
  -/
  rw [← Set.inv_singleton, closure_inv]
  /-
    🎉 no goals
  -/


/-- An induction principle for closure membership. If `p` holds for `1` and all elements of
`k` and their inverse, and is preserved under multiplication, then `p` holds for all elements of
the closure of `k`. -/
@[to_additive (attr := elab_as_elim)
  "An induction principle for additive closure membership. If `p` holds for `0` and all
  elements of `k` and their negation, and is preserved under addition, then `p` holds for all
  elements of the additive closure of `k`."]
theorem closure_induction'' {p : (g : G) → g ∈ closure s → Prop}
    (mem : ∀ x (hx : x ∈ s), p x (subset_closure hx))
    (inv_mem : ∀ x (hx : x ∈ s), p x⁻¹ (inv_mem (subset_closure hx)))
    (one : p 1 (one_mem _))
    (mul : ∀ x y hx hy, p x hx → p y hy → p (x * y) (mul_mem hx hy))
    {x} (h : x ∈ closure s) : p x h :=
  closure_induction_left one (fun x hx y _ hy => mul x y _ _ (mem x hx) hy)
    (fun x hx y _ => mul x⁻¹ y _ _ <| inv_mem x hx) h


/-- An induction principle for elements of `⨆ i, S i`.
If `C` holds for `1` and all elements of `S i` for all `i`, and is preserved under multiplication,
then it holds for all elements of the supremum of `S`. -/
@[to_additive (attr := elab_as_elim) " An induction principle for elements of `⨆ i, S i`.
If `C` holds for `0` and all elements of `S i` for all `i`, and is preserved under addition,
then it holds for all elements of the supremum of `S`. "]
theorem iSup_induction {ι : Sort*} (S : ι → Subgroup G) {C : G → Prop} {x : G} (hx : x ∈ ⨆ i, S i)
    (mem : ∀ (i), ∀ x ∈ S i, C x) (one : C 1) (mul : ∀ x y, C x → C y → C (x * y)) : C x := by
  /-
    G : Type u_2
    inst✝ : Group G
    ι : Sort u_5
    S : ι → Subgroup G
    C : G → Prop
    x : G
    hx : Membership.mem (iSup fun i => S i) x
    mem : ∀ (i : ι) (x : G), Membership.mem (S i) x → C x
    one : C 1
    mul : ∀ (x y : G), C x → C y → C (HMul.hMul x y)
    ⊢ C x
  -/
  rw [iSup_eq_closure] at hx
  induction hx using closure_induction'' with
  | one => exact one
  | mem x hx =>
    obtain ⟨i, hi⟩ := Set.mem_iUnion.mp hx
    exact mem _ _ hi
  | inv_mem x hx =>
    obtain ⟨i, hi⟩ := Set.mem_iUnion.mp hx
    exact mem _ _ (inv_mem hi)
  | mul x y _ _ ihx ihy => exact mul x y ihx ihy


/-- A dependent version of `Subgroup.iSup_induction`. -/
@[to_additive (attr := elab_as_elim) "A dependent version of `AddSubgroup.iSup_induction`. "]
theorem iSup_induction' {ι : Sort*} (S : ι → Subgroup G) {C : ∀ x, (x ∈ ⨆ i, S i) → Prop}
    (hp : ∀ (i), ∀ x (hx : x ∈ S i), C x (mem_iSup_of_mem i hx)) (h1 : C 1 (one_mem _))
    (hmul : ∀ x y hx hy, C x hx → C y hy → C (x * y) (mul_mem ‹_› ‹_›)) {x : G}
    (hx : x ∈ ⨆ i, S i) : C x hx := by
  /-
    G : Type u_2
    inst✝ : Group G
    ι : Sort u_5
    S : ι → Subgroup G
    C : (x : G) → Membership.mem (iSup fun i => S i) x → Prop
    hp : ∀ (i : ι) (x : G) (hx : Membership.mem (S i) x), C x ⋯
    h1 : C 1 ⋯
    hmul : ∀ (x y : G) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membershi …
    x : G
    hx : Membership.mem (iSup fun i => S i) x
    ⊢ C x hx
  -/
  suffices ∃ h, C x h from this.snd
  /-
    G : Type u_2
    inst✝ : Group G
    ι : Sort u_5
    S : ι → Subgroup G
    C : (x : G) → Membership.mem (iSup fun i => S i) x → Prop
    hp : ∀ (i : ι) (x : G) (hx : Membership.mem (S i) x), C x ⋯
    h1 : C 1 ⋯
    hmul : ∀ (x y : G) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membershi …
    x : G
    hx : Membership.mem (iSup fun i => S i) x
    ⊢ Exists fun h => C x h
  -/
  refine iSup_induction S (C := fun x => ∃ h, C x h) hx (fun i x hx => ?_) ?_ fun x y => ?_
    /-
      case refine_1
      G : Type u_2
      inst✝ : Group G
      ι : Sort u_5
      S : ι → Subgroup G
      C : (x : G) → Membership.mem (iSup fun i => S i) x → Prop
      hp : ∀ (i : ι) (x : G) (hx : Membership.mem (S i) x), C x ⋯
      h1 : C 1 ⋯
      hmul : ∀ (x y : G) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membershi …
      x✝ : G
      hx✝ : Membership.mem (iSup fun i => S i) x✝
      i : ι
      x : G
      hx : Membership.mem (S i) x
      ⊢ (fun x => Exists fun h => C x h) x
    -/
  · exact ⟨_, hp i _ hx⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_2
      inst✝ : Group G
      ι : Sort u_5
      S : ι → Subgroup G
      C : (x : G) → Membership.mem (iSup fun i => S i) x → Prop
      hp : ∀ (i : ι) (x : G) (hx : Membership.mem (S i) x), C x ⋯
      h1 : C 1 ⋯
      hmul : ∀ (x y : G) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membershi …
      x : G
      hx : Membership.mem (iSup fun i => S i) x
      ⊢ (fun x => Exists fun h => C x h) 1
    -/
  · exact ⟨_, h1⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      G : Type u_2
      inst✝ : Group G
      ι : Sort u_5
      S : ι → Subgroup G
      C : (x : G) → Membership.mem (iSup fun i => S i) x → Prop
      hp : ∀ (i : ι) (x : G) (hx : Membership.mem (S i) x), C x ⋯
      h1 : C 1 ⋯
      hmul : ∀ (x y : G) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membershi …
      x✝ : G
      hx : Membership.mem (iSup fun i => S i) x✝
      x y : G
      ⊢ (fun x => Exists fun h => C x h) x → (fun x => Exists fun h => C x h) y → (f …
    -/
  · rintro ⟨_, Cx⟩ ⟨_, Cy⟩
    /-
      case refine_3.intro.intro
      G : Type u_2
      inst✝ : Group G
      ι : Sort u_5
      S : ι → Subgroup G
      C : (x : G) → Membership.mem (iSup fun i => S i) x → Prop
      hp : ∀ (i : ι) (x : G) (hx : Membership.mem (S i) x), C x ⋯
      h1 : C 1 ⋯
      hmul : ∀ (x y : G) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membershi …
      x✝ : G
      hx : Membership.mem (iSup fun i => S i) x✝
      x y : G
      w✝¹ : Membership.mem (iSup fun i => S i) x
      Cx : C x w✝¹
      w✝ : Membership.mem (iSup fun i => S i) y
      Cy : C y w✝
      ⊢ Exists fun h => C (HMul.hMul x y) h
    -/
    exact ⟨_, hmul _ _ _ _ Cx Cy⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem closure_mul_le (S T : Set G) : closure (S * T) ≤ closure S ⊔ closure T :=
  sInf_le fun _x ⟨_s, hs, _t, ht, hx⟩ => hx ▸
    (closure S ⊔ closure T).mul_mem (SetLike.le_def.mp le_sup_left <| subset_closure hs)
      (SetLike.le_def.mp le_sup_right <| subset_closure ht)


@[to_additive]
lemma closure_pow_le : ∀ {n}, n ≠ 0 → closure (s ^ n) ≤ closure s
               /-
                 G : Type u_2
                 inst✝ : Group G
                 s : Set G
                 x✝ : Ne 1 0
                 ⊢ LE.le (Subgroup.closure (HPow.hPow s 1)) (Subgroup.closure s)
               -/
  | 1, _ => by simp
               /-
                 🎉 no goals
               -/
  | n + 2, _ =>
    calc
      closure (s ^ (n + 2))
                                          /-
                                            G : Type u_2
                                            inst✝ : Group G
                                            s : Set G
                                            n : Nat
                                            x✝ : Ne (HAdd.hAdd n 2) 0
                                            ⊢ Eq (Subgroup.closure (HPow.hPow s (HAdd.hAdd n 2))) (Subgroup.closure (HMul. …
                                          -/
      _ = closure (s ^ (n + 1) * s) := by rw [pow_succ]
                                          /-
                                            🎉 no goals
                                          -/
      _ ≤ closure (s ^ (n + 1)) ⊔ closure s := closure_mul_le ..
                                      /-
                                        G : Type u_2
                                        inst✝ : Group G
                                        s : Set G
                                        n : Nat
                                        x✝ : Ne (HAdd.hAdd n 2) 0
                                        ⊢ LE.le (Max.max (Subgroup.closure (HPow.hPow s (HAdd.hAdd n 1))) (Subgroup.cl …
                                      -/
      _ ≤ closure s ⊔ closure s := by gcongr ?_ ⊔ _; exact closure_pow_le n.succ_ne_zero
                                                     /-
                                                       🎉 no goals
                                                     -/
      _ = closure s := sup_idem _


@[to_additive]
lemma closure_pow {n : ℕ} (hs : 1 ∈ s) (hn : n ≠ 0) : closure (s ^ n) = closure s :=
                                     /-
                                       G : Type u_2
                                       inst✝ : Group G
                                       s : Set G
                                       n : Nat
                                       hs : Membership.mem s 1
                                       hn : Ne n 0
                                       ⊢ LE.le (Subgroup.closure s) (Subgroup.closure (HPow.hPow s n))
                                     -/
  (closure_pow_le hn).antisymm <| by gcongr; exact subset_pow hs hn
                                             /-
                                               🎉 no goals
                                             -/


@[to_additive]
theorem sup_eq_closure_mul (H K : Subgroup G) : H ⊔ K = closure ((H : Set G) * (K : Set G)) :=
  le_antisymm
    (sup_le (fun h hh => subset_closure ⟨h, hh, 1, K.one_mem, mul_one h⟩) fun k hk =>
      subset_closure ⟨1, H.one_mem, k, hk, one_mul k⟩)
                                      /-
                                        G : Type u_2
                                        inst✝ : Group G
                                        H K : Subgroup G
                                        ⊢ LE.le (Max.max (Subgroup.closure ↑H) (Subgroup.closure ↑K)) (Max.max H K)
                                      -/
    ((closure_mul_le _ _).trans <| by rw [closure_eq, closure_eq])
                                      /-
                                        🎉 no goals
                                      -/


@[to_additive]
theorem set_mul_normal_comm (s : Set G) (N : Subgroup G) [hN : N.Normal] :
    s * (N : Set G) = (N : Set G) * s := by
  /-
    G : Type u_2
    inst✝ : Group G
    s : Set G
    N : Subgroup G
    hN : N.Normal
    ⊢ Eq (HMul.hMul s ↑N) (HMul.hMul (↑N) s)
  -/
  rw [← iUnion_mul_left_image, ← iUnion_mul_right_image]
  /-
    G : Type u_2
    inst✝ : Group G
    s : Set G
    N : Subgroup G
    hN : N.Normal
    ⊢ Eq (Set.iUnion fun a => Set.iUnion fun h => Set.image (fun x => HMul.hMul a  …
  -/
  simp only [image_mul_left, image_mul_right, Set.preimage, SetLike.mem_coe, hN.mem_comm_iff]
  /-
    🎉 no goals
  -/


/-- The carrier of `H ⊔ N` is just `↑H * ↑N` (pointwise set product) when `N` is normal. -/
@[to_additive "The carrier of `H ⊔ N` is just `↑H + ↑N` (pointwise set addition)
when `N` is normal."]
theorem mul_normal (H N : Subgroup G) [hN : N.Normal] : (↑(H ⊔ N) : Set G) = H * N := by
  /-
    G : Type u_2
    inst✝ : Group G
    H N : Subgroup G
    hN : N.Normal
    ⊢ Eq (↑(Max.max H N)) (HMul.hMul ↑H ↑N)
  -/
  rw [sup_eq_closure_mul]
  /-
    G : Type u_2
    inst✝ : Group G
    H N : Subgroup G
    hN : N.Normal
    ⊢ Eq (↑(Subgroup.closure (HMul.hMul ↑H ↑N))) (HMul.hMul ↑H ↑N)
  -/
  refine Set.Subset.antisymm (fun x hx => ?_) subset_closure
  induction hx using closure_induction'' with
  | one => exact ⟨1, one_mem _, 1, one_mem _, mul_one 1⟩
  | mem _ hx => exact hx
  | inv_mem x hx =>
    obtain ⟨x, hx, y, hy, rfl⟩ := hx
    simpa only [mul_inv_rev, mul_assoc, inv_inv, inv_mul_cancel_left]
      using mul_mem_mul (inv_mem hx) (hN.conj_mem _ (inv_mem hy) x)
  | mul x' x' _ _ hx hx' =>
    obtain ⟨x, hx, y, hy, rfl⟩ := hx
    obtain ⟨x', hx', y', hy', rfl⟩ := hx'
    refine ⟨x * x', mul_mem hx hx', x'⁻¹ * y * x' * y', mul_mem ?_ hy', ?_⟩
    · simpa using hN.conj_mem _ hy x'⁻¹
    · simp only [mul_assoc, mul_inv_cancel_left]


/-- The carrier of `N ⊔ H` is just `↑N * ↑H` (pointwise set product) when `N` is normal. -/
@[to_additive "The carrier of `N ⊔ H` is just `↑N + ↑H` (pointwise set addition)
when `N` is normal."]
theorem normal_mul (N H : Subgroup G) [N.Normal] : (↑(N ⊔ H) : Set G) = N * H := by
  /-
    G : Type u_2
    inst✝¹ : Group G
    N H : Subgroup G
    inst✝ : N.Normal
    ⊢ Eq (↑(Max.max N H)) (HMul.hMul ↑N ↑H)
  -/
  rw [← set_mul_normal_comm, sup_comm, mul_normal]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_inf_assoc (A B C : Subgroup G) (h : A ≤ C) :
    (A : Set G) * ↑(B ⊓ C) = (A : Set G) * (B : Set G) ∩ C := by
  /-
    G : Type u_2
    inst✝ : Group G
    A B C : Subgroup G
    h : LE.le A C
    ⊢ Eq (HMul.hMul ↑A ↑(Min.min B C)) (Inter.inter (HMul.hMul ↑A ↑B) ↑C)
  -/
  ext
  /-
    case h
    G : Type u_2
    inst✝ : Group G
    A B C : Subgroup G
    h : LE.le A C
    x✝ : G
    ⊢ Iff (Membership.mem (HMul.hMul ↑A ↑(Min.min B C)) x✝) (Membership.mem (Inter …
  -/
  simp only [coe_inf, Set.mem_mul, Set.mem_inter_iff]
  /-
    case h
    G : Type u_2
    inst✝ : Group G
    A B C : Subgroup G
    h : LE.le A C
    x✝ : G
    ⊢ Iff (Exists fun x => And (Membership.mem (↑A) x) (Exists fun y => And (And ( …
  -/
  constructor
    /-
      case h.mp
      G : Type u_2
      inst✝ : Group G
      A B C : Subgroup G
      h : LE.le A C
      x✝ : G
      ⊢ (Exists fun x => And (Membership.mem (↑A) x) (Exists fun y => And (And (Memb …
    -/
  · rintro ⟨y, hy, z, ⟨hzB, hzC⟩, rfl⟩
    /-
      case h.mp.intro.intro.intro.intro.intro
      G : Type u_2
      inst✝ : Group G
      A B C : Subgroup G
      h : LE.le A C
      y : G
      hy : Membership.mem (↑A) y
      z : G
      hzB : Membership.mem (↑B) z
      hzC : Membership.mem (↑C) z
      ⊢ And (Exists fun x => And (Membership.mem (↑A) x) (Exists fun y_1 => And (Mem …
    -/
    refine ⟨?_, mul_mem (h hy) hzC⟩
    /-
      case h.mp.intro.intro.intro.intro.intro
      G : Type u_2
      inst✝ : Group G
      A B C : Subgroup G
      h : LE.le A C
      y : G
      hy : Membership.mem (↑A) y
      z : G
      hzB : Membership.mem (↑B) z
      hzC : Membership.mem (↑C) z
      ⊢ Exists fun x => And (Membership.mem (↑A) x) (Exists fun y_1 => And (Membersh …
    -/
    exact ⟨y, hy, z, hzB, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case h.mpr
    G : Type u_2
    inst✝ : Group G
    A B C : Subgroup G
    h : LE.le A C
    x✝ : G
    ⊢ And (Exists fun x => And (Membership.mem (↑A) x) (Exists fun y => And (Membe …
  -/
  rintro ⟨⟨y, hy, z, hz, rfl⟩, hyz⟩
  /-
    case h.mpr.intro.intro.intro.intro.intro
    G : Type u_2
    inst✝ : Group G
    A B C : Subgroup G
    h : LE.le A C
    y : G
    hy : Membership.mem (↑A) y
    z : G
    hz : Membership.mem (↑B) z
    hyz : Membership.mem (↑C) (HMul.hMul y z)
    ⊢ Exists fun x => And (Membership.mem (↑A) x) (Exists fun y_1 => And (And (Mem …
  -/
  refine ⟨y, hy, z, ⟨hz, ?_⟩, rfl⟩
  /-
    case h.mpr.intro.intro.intro.intro.intro
    G : Type u_2
    inst✝ : Group G
    A B C : Subgroup G
    h : LE.le A C
    y : G
    hy : Membership.mem (↑A) y
    z : G
    hz : Membership.mem (↑B) z
    hyz : Membership.mem (↑C) (HMul.hMul y z)
    ⊢ Membership.mem (↑C) z
  -/
  suffices y⁻¹ * (y * z) ∈ C by simpa
  /-
    case h.mpr.intro.intro.intro.intro.intro
    G : Type u_2
    inst✝ : Group G
    A B C : Subgroup G
    h : LE.le A C
    y : G
    hy : Membership.mem (↑A) y
    z : G
    hz : Membership.mem (↑B) z
    hyz : Membership.mem (↑C) (HMul.hMul y z)
    ⊢ Membership.mem C (HMul.hMul (Inv.inv y) (HMul.hMul y z))
  -/
  exact mul_mem (inv_mem (h hy)) hyz
  /-
    🎉 no goals
  -/


@[to_additive]
theorem inf_mul_assoc (A B C : Subgroup G) (h : C ≤ A) :
    ((A ⊓ B : Subgroup G) : Set G) * C = (A : Set G) ∩ (↑B * ↑C) := by
  /-
    G : Type u_2
    inst✝ : Group G
    A B C : Subgroup G
    h : LE.le C A
    ⊢ Eq (HMul.hMul ↑(Min.min A B) ↑C) (Inter.inter (↑A) (HMul.hMul ↑B ↑C))
  -/
  ext
  /-
    case h
    G : Type u_2
    inst✝ : Group G
    A B C : Subgroup G
    h : LE.le C A
    x✝ : G
    ⊢ Iff (Membership.mem (HMul.hMul ↑(Min.min A B) ↑C) x✝) (Membership.mem (Inter …
  -/
  simp only [coe_inf, Set.mem_mul, Set.mem_inter_iff]
  /-
    case h
    G : Type u_2
    inst✝ : Group G
    A B C : Subgroup G
    h : LE.le C A
    x✝ : G
    ⊢ Iff (Exists fun x => And (And (Membership.mem (↑A) x) (Membership.mem (↑B) x …
  -/
  constructor
    /-
      case h.mp
      G : Type u_2
      inst✝ : Group G
      A B C : Subgroup G
      h : LE.le C A
      x✝ : G
      ⊢ (Exists fun x => And (And (Membership.mem (↑A) x) (Membership.mem (↑B) x)) ( …
    -/
  · rintro ⟨y, ⟨hyA, hyB⟩, z, hz, rfl⟩
    /-
      case h.mp.intro.intro.intro.intro.intro
      G : Type u_2
      inst✝ : Group G
      A B C : Subgroup G
      h : LE.le C A
      y : G
      hyA : Membership.mem (↑A) y
      hyB : Membership.mem (↑B) y
      z : G
      hz : Membership.mem (↑C) z
      ⊢ And (Membership.mem (↑A) (HMul.hMul y z)) (Exists fun x => And (Membership.m …
    -/
    refine ⟨A.mul_mem hyA (h hz), ?_⟩
    /-
      case h.mp.intro.intro.intro.intro.intro
      G : Type u_2
      inst✝ : Group G
      A B C : Subgroup G
      h : LE.le C A
      y : G
      hyA : Membership.mem (↑A) y
      hyB : Membership.mem (↑B) y
      z : G
      hz : Membership.mem (↑C) z
      ⊢ Exists fun x => And (Membership.mem (↑B) x) (Exists fun y_1 => And (Membersh …
    -/
    exact ⟨y, hyB, z, hz, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case h.mpr
    G : Type u_2
    inst✝ : Group G
    A B C : Subgroup G
    h : LE.le C A
    x✝ : G
    ⊢ And (Membership.mem (↑A) x✝) (Exists fun x => And (Membership.mem (↑B) x) (E …
  -/
  rintro ⟨hyz, y, hy, z, hz, rfl⟩
  /-
    case h.mpr.intro.intro.intro.intro.intro
    G : Type u_2
    inst✝ : Group G
    A B C : Subgroup G
    h : LE.le C A
    y : G
    hy : Membership.mem (↑B) y
    z : G
    hz : Membership.mem (↑C) z
    hyz : Membership.mem (↑A) (HMul.hMul y z)
    ⊢ Exists fun x => And (And (Membership.mem (↑A) x) (Membership.mem (↑B) x)) (E …
  -/
  refine ⟨y, ⟨?_, hy⟩, z, hz, rfl⟩
  /-
    case h.mpr.intro.intro.intro.intro.intro
    G : Type u_2
    inst✝ : Group G
    A B C : Subgroup G
    h : LE.le C A
    y : G
    hy : Membership.mem (↑B) y
    z : G
    hz : Membership.mem (↑C) z
    hyz : Membership.mem (↑A) (HMul.hMul y z)
    ⊢ Membership.mem (↑A) y
  -/
  suffices y * z * z⁻¹ ∈ A by simpa
  /-
    case h.mpr.intro.intro.intro.intro.intro
    G : Type u_2
    inst✝ : Group G
    A B C : Subgroup G
    h : LE.le C A
    y : G
    hy : Membership.mem (↑B) y
    z : G
    hz : Membership.mem (↑C) z
    hyz : Membership.mem (↑A) (HMul.hMul y z)
    ⊢ Membership.mem A (HMul.hMul (HMul.hMul y z) (Inv.inv z))
  -/
  exact mul_mem hyz (inv_mem (h hz))
  /-
    🎉 no goals
  -/


@[to_additive]
instance sup_normal (H K : Subgroup G) [hH : H.Normal] [hK : K.Normal] : (H ⊔ K).Normal where
  conj_mem n hmem g := by
    /-
      α : Type u_1
      G : Type u_2
      A : Type u_3
      S : Type u_4
      inst✝¹ : Group G
      inst✝ : AddGroup A
      s : Set G
      H K : Subgroup G
      hH : H.Normal
      hK : K.Normal
      n : G
      hmem : Membership.mem (Max.max H K) n
      g : G
      ⊢ Membership.mem (Max.max H K) (HMul.hMul (HMul.hMul g n) (Inv.inv g))
    -/
    rw [← SetLike.mem_coe, normal_mul] at hmem ⊢
    /-
      α : Type u_1
      G : Type u_2
      A : Type u_3
      S : Type u_4
      inst✝¹ : Group G
      inst✝ : AddGroup A
      s : Set G
      H K : Subgroup G
      hH : H.Normal
      hK : K.Normal
      n : G
      hmem : Membership.mem (HMul.hMul ↑H ↑K) n
      g : G
      ⊢ Membership.mem (HMul.hMul ↑H ↑K) (HMul.hMul (HMul.hMul g n) (Inv.inv g))
    -/
    rcases hmem with ⟨h, hh, k, hk, rfl⟩
    /-
      case intro.intro.intro.intro
      α : Type u_1
      G : Type u_2
      A : Type u_3
      S : Type u_4
      inst✝¹ : Group G
      inst✝ : AddGroup A
      s : Set G
      H K : Subgroup G
      hH : H.Normal
      hK : K.Normal
      g h : G
      hh : Membership.mem (↑H) h
      k : G
      hk : Membership.mem (↑K) k
      ⊢ Membership.mem (HMul.hMul ↑H ↑K) (HMul.hMul (HMul.hMul g ((fun x1 x2 => HMul …
    -/
    refine ⟨g * h * g⁻¹, hH.conj_mem h hh g, g * k * g⁻¹, hK.conj_mem k hk g, ?_⟩
    /-
      case intro.intro.intro.intro
      α : Type u_1
      G : Type u_2
      A : Type u_3
      S : Type u_4
      inst✝¹ : Group G
      inst✝ : AddGroup A
      s : Set G
      H K : Subgroup G
      hH : H.Normal
      hK : K.Normal
      g h : G
      hh : Membership.mem (↑H) h
      k : G
      hk : Membership.mem (↑K) k
      ⊢ Eq ((fun x1 x2 => HMul.hMul x1 x2) (HMul.hMul (HMul.hMul g h) (Inv.inv g)) ( …
    -/
    simp only [mul_assoc, inv_mul_cancel_left]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem smul_mem_of_mem_closure_of_mem {X : Type*} [MulAction G X] {s : Set G} {t : Set X}
    (hs : ∀ g ∈ s, g⁻¹ ∈ s) (hst : ∀ᵉ (g ∈ s) (x ∈ t), g • x ∈ t) {g : G}
    (hg : g ∈ Subgroup.closure s) {x : X} (hx : x ∈ t) : g • x ∈ t := by
  induction hg using Subgroup.closure_induction'' generalizing x with
  | one => simpa
  | mem g' hg' => exact hst g' hg' x hx
  | inv_mem g' hg' => exact hst g'⁻¹ (hs g' hg') x hx
  | mul _ _ _ _ h₁ h₂ => rw [mul_smul]; exact h₁ (h₂ hx)


@[to_additive]
theorem smul_opposite_image_mul_preimage' (g : G) (h : Gᵐᵒᵖ) (s : Set G) :
    (fun y => h • y) '' ((g * ·) ⁻¹' s) = (g * ·) ⁻¹' ((fun y => h • y) '' s) := by
  /-
    G : Type u_2
    inst✝ : Group G
    g : G
    h : MulOpposite G
    s : Set G
    ⊢ Eq (Set.image (fun y => HSMul.hSMul h y) (Set.preimage (fun x => HMul.hMul g …
  -/
  simp [preimage_preimage, mul_assoc]
  /-
    🎉 no goals
  -/

-- Porting note: deprecate?

@[to_additive]
theorem smul_opposite_image_mul_preimage {H : Subgroup G} (g : G) (h : H.op) (s : Set G) :
    (fun y => h • y) '' ((g * ·) ⁻¹' s) = (g * ·) ⁻¹' ((fun y => h • y) '' s) :=
  smul_opposite_image_mul_preimage' g h s


/-- The action on a subgroup corresponding to applying the action to every element.

This is available as an instance in the `Pointwise` locale. -/
protected def pointwiseMulAction : MulAction α (Subgroup G) where
  smul a S := S.map (MulDistribMulAction.toMonoidEnd _ _ a)
  one_smul S := by
    /-
      α : Type u_1
      G : Type u_2
      A : Type u_3
      S✝ : Type u_4
      inst✝³ : Group G
      inst✝² : AddGroup A
      s : Set G
      inst✝¹ : Monoid α
      inst✝ : MulDistribMulAction α G
      S : Subgroup G
      ⊢ Eq (HSMul.hSMul 1 S) S
    -/
    change S.map _ = S
    /-
      α : Type u_1
      G : Type u_2
      A : Type u_3
      S✝ : Type u_4
      inst✝³ : Group G
      inst✝² : AddGroup A
      s : Set G
      inst✝¹ : Monoid α
      inst✝ : MulDistribMulAction α G
      S : Subgroup G
      ⊢ Eq (Subgroup.map ((MulDistribMulAction.toMonoidEnd α G) 1) S) S
    -/
    simpa only [map_one] using S.map_id
    /-
      🎉 no goals
    -/
  mul_smul _ _ S :=
    (congr_arg (fun f : Monoid.End G => S.map f) (MonoidHom.map_mul _ _ _)).trans
      (S.map_map _ _).symm


theorem pointwise_smul_def {a : α} (S : Subgroup G) :
    a • S = S.map (MulDistribMulAction.toMonoidEnd _ _ a) :=
  rfl


@[simp]
theorem coe_pointwise_smul (a : α) (S : Subgroup G) : ↑(a • S) = a • (S : Set G) :=
  rfl


@[simp]
theorem pointwise_smul_toSubmonoid (a : α) (S : Subgroup G) :
    (a • S).toSubmonoid = a • S.toSubmonoid :=
  rfl


theorem smul_mem_pointwise_smul (m : G) (a : α) (S : Subgroup G) : m ∈ S → a • m ∈ a • S :=
  (Set.smul_mem_smul_set : _ → _ ∈ a • (S : Set G))


instance : CovariantClass α (Subgroup G) HSMul.hSMul LE.le :=
  ⟨fun _ _ => image_subset _⟩


theorem mem_smul_pointwise_iff_exists (m : G) (a : α) (S : Subgroup G) :
    m ∈ a • S ↔ ∃ s : G, s ∈ S ∧ a • s = m :=
  (Set.mem_smul_set : m ∈ a • (S : Set G) ↔ _)


@[simp]
theorem smul_bot (a : α) : a • (⊥ : Subgroup G) = ⊥ :=
  map_bot _


theorem smul_sup (a : α) (S T : Subgroup G) : a • (S ⊔ T) = a • S ⊔ a • T :=
  map_sup _ _ _


theorem smul_closure (a : α) (s : Set G) : a • closure s = closure (a • s) :=
  MonoidHom.map_closure _ _


instance pointwise_isCentralScalar [MulDistribMulAction αᵐᵒᵖ G] [IsCentralScalar α G] :
    IsCentralScalar α (Subgroup G) :=
  ⟨fun _ S => (congr_arg fun f => S.map f) <| MonoidHom.ext <| op_smul_eq_smul _⟩


theorem conj_smul_le_of_le {P H : Subgroup G} (hP : P ≤ H) (h : H) :
    MulAut.conj (h : G) • P ≤ H := by
  /-
    G : Type u_2
    inst✝ : Group G
    P H : Subgroup G
    hP : LE.le P H
    h : Subtype fun x => Membership.mem H x
    ⊢ LE.le (HSMul.hSMul (MulAut.conj ↑h) P) H
  -/
  rintro - ⟨g, hg, rfl⟩
  /-
    case intro.intro
    G : Type u_2
    inst✝ : Group G
    P H : Subgroup G
    hP : LE.le P H
    h : Subtype fun x => Membership.mem H x
    g : G
    hg : Membership.mem (↑P) g
    ⊢ Membership.mem H (((MulDistribMulAction.toMonoidEnd (MulAut G) G) (MulAut.co …
  -/
  exact H.mul_mem (H.mul_mem h.2 (hP hg)) (H.inv_mem h.2)
  /-
    🎉 no goals
  -/


theorem conj_smul_subgroupOf {P H : Subgroup G} (hP : P ≤ H) (h : H) :
    MulAut.conj h • P.subgroupOf H = (MulAut.conj (h : G) • P).subgroupOf H := by
  /-
    G : Type u_2
    inst✝ : Group G
    P H : Subgroup G
    hP : LE.le P H
    h : Subtype fun x => Membership.mem H x
    ⊢ Eq (HSMul.hSMul (MulAut.conj h) (P.subgroupOf H)) ((HSMul.hSMul (MulAut.conj …
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      G : Type u_2
      inst✝ : Group G
      P H : Subgroup G
      hP : LE.le P H
      h : Subtype fun x => Membership.mem H x
      ⊢ LE.le (HSMul.hSMul (MulAut.conj h) (P.subgroupOf H)) ((HSMul.hSMul (MulAut.c …
    -/
  · rintro - ⟨g, hg, rfl⟩
    /-
      case refine_1.intro.intro
      G : Type u_2
      inst✝ : Group G
      P H : Subgroup G
      hP : LE.le P H
      h g : Subtype fun x => Membership.mem H x
      hg : Membership.mem (↑(P.subgroupOf H)) g
      ⊢ Membership.mem ((HSMul.hSMul (MulAut.conj ↑h) P).subgroupOf H) (((MulDistrib …
    -/
    exact ⟨g, hg, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_2
      inst✝ : Group G
      P H : Subgroup G
      hP : LE.le P H
      h : Subtype fun x => Membership.mem H x
      ⊢ LE.le ((HSMul.hSMul (MulAut.conj ↑h) P).subgroupOf H) (HSMul.hSMul (MulAut.c …
    -/
  · rintro p ⟨g, hg, hp⟩
    /-
      case refine_2.intro.intro
      G : Type u_2
      inst✝ : Group G
      P H : Subgroup G
      hP : LE.le P H
      h p : Subtype fun x => Membership.mem H x
      g : G
      hg : Membership.mem (↑P) g
      hp : Eq (((MulDistribMulAction.toMonoidEnd (MulAut G) G) (MulAut.conj ↑h)) g)  …
      ⊢ Membership.mem (HSMul.hSMul (MulAut.conj h) (P.subgroupOf H)) p
    -/
    exact ⟨⟨g, hP hg⟩, hg, Subtype.ext hp⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem smul_mem_pointwise_smul_iff {a : α} {S : Subgroup G} {x : G} : a • x ∈ a • S ↔ x ∈ S :=
  smul_mem_smul_set_iff


theorem mem_pointwise_smul_iff_inv_smul_mem {a : α} {S : Subgroup G} {x : G} :
    x ∈ a • S ↔ a⁻¹ • x ∈ S :=
  mem_smul_set_iff_inv_smul_mem


theorem mem_inv_pointwise_smul_iff {a : α} {S : Subgroup G} {x : G} : x ∈ a⁻¹ • S ↔ a • x ∈ S :=
  mem_inv_smul_set_iff


@[simp]
theorem pointwise_smul_le_pointwise_smul_iff {a : α} {S T : Subgroup G} : a • S ≤ a • T ↔ S ≤ T :=
  set_smul_subset_set_smul_iff


theorem pointwise_smul_subset_iff {a : α} {S T : Subgroup G} : a • S ≤ T ↔ S ≤ a⁻¹ • T :=
  set_smul_subset_iff


theorem subset_pointwise_smul_iff {a : α} {S T : Subgroup G} : S ≤ a • T ↔ a⁻¹ • S ≤ T :=
  subset_set_smul_iff


@[simp]
theorem smul_inf (a : α) (S T : Subgroup G) : a • (S ⊓ T) = a • S ⊓ a • T := by
  /-
    α : Type u_1
    G : Type u_2
    inst✝² : Group G
    inst✝¹ : Group α
    inst✝ : MulDistribMulAction α G
    a : α
    S T : Subgroup G
    ⊢ Eq (HSMul.hSMul a (Min.min S T)) (Min.min (HSMul.hSMul a S) (HSMul.hSMul a T))
  -/
  simp [SetLike.ext_iff, mem_pointwise_smul_iff_inv_smul_mem]
  /-
    🎉 no goals
  -/


/-- Applying a `MulDistribMulAction` results in an isomorphic subgroup -/
@[simps!]
def equivSMul (a : α) (H : Subgroup G) : H ≃* (a • H : Subgroup G) :=
  (MulDistribMulAction.toMulEquiv G a).subgroupMap H


theorem subgroup_mul_singleton {H : Subgroup G} {h : G} (hh : h ∈ H) : (H : Set G) * {h} = H := by
  /-
    G : Type u_2
    inst✝ : Group G
    H : Subgroup G
    h : G
    hh : Membership.mem H h
    ⊢ Eq (HMul.hMul (↑H) (Singleton.singleton h)) ↑H
  -/
  simp [preimage, mul_mem_cancel_right (inv_mem hh)]
  /-
    🎉 no goals
  -/


theorem singleton_mul_subgroup {H : Subgroup G} {h : G} (hh : h ∈ H) : {h} * (H : Set G) = H := by
  /-
    G : Type u_2
    inst✝ : Group G
    H : Subgroup G
    h : G
    hh : Membership.mem H h
    ⊢ Eq (HMul.hMul (Singleton.singleton h) ↑H) ↑H
  -/
  simp [preimage, mul_mem_cancel_left (inv_mem hh)]
  /-
    🎉 no goals
  -/


theorem Normal.conjAct {H : Subgroup G} (hH : H.Normal) (g : ConjAct G) : g • H = H :=
  have : ∀ g : ConjAct G, g • H ≤ H :=
    fun _ => map_le_iff_le_comap.2 fun _ h => hH.conj_mem _ h _
  (this g).antisymm <| (smul_inv_smul g H).symm.trans_le (map_mono <| this _)


@[simp]
theorem smul_normal (g : G) (H : Subgroup G) [h : Normal H] : MulAut.conj g • H = H :=
  h.conjAct g


theorem Normal.of_conjugate_fixed {H : Subgroup G} (h : ∀ g : G, (MulAut.conj g) • H = H) :
    H.Normal := by
  /-
    G : Type u_2
    inst✝ : Group G
    H : Subgroup G
    h : ∀ (g : G), Eq (HSMul.hSMul (MulAut.conj g) H) H
    ⊢ H.Normal
  -/
  constructor
  /-
    case conj_mem
    G : Type u_2
    inst✝ : Group G
    H : Subgroup G
    h : ∀ (g : G), Eq (HSMul.hSMul (MulAut.conj g) H) H
    ⊢ ∀ (n : G), Membership.mem H n → ∀ (g : G), Membership.mem H (HMul.hMul (HMul …
  -/
  intro n hn g
  rw [← h g, Subgroup.mem_pointwise_smul_iff_inv_smul_mem, ← map_inv, MulAut.smul_def,
    MulAut.conj_apply, inv_inv, mul_assoc, mul_assoc, inv_mul_cancel, mul_one,
    ← mul_assoc, inv_mul_cancel, one_mul]
  /-
    case conj_mem
    G : Type u_2
    inst✝ : Group G
    H : Subgroup G
    h : ∀ (g : G), Eq (HSMul.hSMul (MulAut.conj g) H) H
    n : G
    hn : Membership.mem H n
    g : G
    ⊢ Membership.mem H n
  -/
  exact hn
  /-
    🎉 no goals
  -/


theorem normalCore_eq_iInf_conjAct (H : Subgroup G) :
    H.normalCore = ⨅ (g : ConjAct G), g • H := by
  /-
    G : Type u_2
    inst✝ : Group G
    H : Subgroup G
    ⊢ Eq H.normalCore (iInf fun g => HSMul.hSMul g H)
  -/
  ext g
  /-
    case h
    G : Type u_2
    inst✝ : Group G
    H : Subgroup G
    g : G
    ⊢ Iff (Membership.mem H.normalCore g) (Membership.mem (iInf fun g => HSMul.hSM …
  -/
  simp only [Subgroup.normalCore, Subgroup.mem_iInf, Subgroup.mem_pointwise_smul_iff_inv_smul_mem]
  /-
    case h
    G : Type u_2
    inst✝ : Group G
    H : Subgroup G
    g : G
    ⊢ Iff (Membership.mem { carrier := setOf fun a => ∀ (b : G), Membership.mem H  …
  -/
  refine ⟨fun h x ↦ h x⁻¹, fun h x ↦ ?_⟩
  /-
    case h
    G : Type u_2
    inst✝ : Group G
    H : Subgroup G
    g : G
    h : ∀ (i : ConjAct G), Membership.mem H (HSMul.hSMul (Inv.inv i) g)
    x : G
    ⊢ Membership.mem H (HMul.hMul (HMul.hMul x g) (Inv.inv x))
  -/
  simpa only [ConjAct.toConjAct_inv, inv_inv] using h x⁻¹
  /-
    🎉 no goals
  -/


@[simp]
theorem smul_mem_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) (S : Subgroup G) (x : G) :
    a • x ∈ a • S ↔ x ∈ S :=
  smul_mem_smul_set_iff₀ ha (S : Set G) x


theorem mem_pointwise_smul_iff_inv_smul_mem₀ {a : α} (ha : a ≠ 0) (S : Subgroup G) (x : G) :
    x ∈ a • S ↔ a⁻¹ • x ∈ S :=
  mem_smul_set_iff_inv_smul_mem₀ ha (S : Set G) x


theorem mem_inv_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) (S : Subgroup G) (x : G) :
    x ∈ a⁻¹ • S ↔ a • x ∈ S :=
  mem_inv_smul_set_iff₀ ha (S : Set G) x


@[simp]
theorem pointwise_smul_le_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) {S T : Subgroup G} :
    a • S ≤ a • T ↔ S ≤ T :=
  set_smul_subset_set_smul_iff₀ ha


theorem pointwise_smul_le_iff₀ {a : α} (ha : a ≠ 0) {S T : Subgroup G} : a • S ≤ T ↔ S ≤ a⁻¹ • T :=
  set_smul_subset_iff₀ ha


theorem le_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) {S T : Subgroup G} : S ≤ a • T ↔ a⁻¹ • S ≤ T :=
  subset_set_smul_iff₀ ha


/-- The action on an additive subgroup corresponding to applying the action to every element.

This is available as an instance in the `Pointwise` locale. -/
protected def pointwiseMulAction : MulAction α (AddSubgroup A) where
  smul a S := S.map (DistribMulAction.toAddMonoidEnd _ _ a)
  one_smul S := by
    /-
      α : Type u_1
      G : Type u_2
      A : Type u_3
      S✝ : Type u_4
      inst✝³ : Group G
      inst✝² : AddGroup A
      s : Set G
      inst✝¹ : Monoid α
      inst✝ : DistribMulAction α A
      S : AddSubgroup A
      ⊢ Eq (HSMul.hSMul 1 S) S
    -/
    change S.map _ = S
    /-
      α : Type u_1
      G : Type u_2
      A : Type u_3
      S✝ : Type u_4
      inst✝³ : Group G
      inst✝² : AddGroup A
      s : Set G
      inst✝¹ : Monoid α
      inst✝ : DistribMulAction α A
      S : AddSubgroup A
      ⊢ Eq (AddSubgroup.map ((DistribMulAction.toAddMonoidEnd α A) 1) S) S
    -/
    simpa only [map_one] using S.map_id
    /-
      🎉 no goals
    -/
  mul_smul _ _ S :=
    (congr_arg (fun f : AddMonoid.End A => S.map f) (MonoidHom.map_mul _ _ _)).trans
      (S.map_map _ _).symm


theorem pointwise_smul_def {a : α} (S : AddSubgroup A) :
    a • S = S.map (DistribMulAction.toAddMonoidEnd _ _ a) :=
  rfl


@[simp]
theorem coe_pointwise_smul (a : α) (S : AddSubgroup A) : ↑(a • S) = a • (S : Set A) :=
  rfl


@[simp]
theorem pointwise_smul_toAddSubmonoid (a : α) (S : AddSubgroup A) :
    (a • S).toAddSubmonoid = a • S.toAddSubmonoid :=
  rfl


theorem smul_mem_pointwise_smul (m : A) (a : α) (S : AddSubgroup A) : m ∈ S → a • m ∈ a • S :=
  (Set.smul_mem_smul_set : _ → _ ∈ a • (S : Set A))


theorem mem_smul_pointwise_iff_exists (m : A) (a : α) (S : AddSubgroup A) :
    m ∈ a • S ↔ ∃ s : A, s ∈ S ∧ a • s = m :=
  (Set.mem_smul_set : m ∈ a • (S : Set A) ↔ _)


instance pointwise_isCentralScalar [DistribMulAction αᵐᵒᵖ A] [IsCentralScalar α A] :
    IsCentralScalar α (AddSubgroup A) :=
  ⟨fun _ S => (congr_arg fun f => S.map f) <| AddMonoidHom.ext <| op_smul_eq_smul _⟩


@[simp]
theorem smul_mem_pointwise_smul_iff {a : α} {S : AddSubgroup A} {x : A} : a • x ∈ a • S ↔ x ∈ S :=
  smul_mem_smul_set_iff


theorem mem_pointwise_smul_iff_inv_smul_mem {a : α} {S : AddSubgroup A} {x : A} :
    x ∈ a • S ↔ a⁻¹ • x ∈ S :=
  mem_smul_set_iff_inv_smul_mem


theorem mem_inv_pointwise_smul_iff {a : α} {S : AddSubgroup A} {x : A} : x ∈ a⁻¹ • S ↔ a • x ∈ S :=
  mem_inv_smul_set_iff


@[simp]
theorem pointwise_smul_le_pointwise_smul_iff {a : α} {S T : AddSubgroup A} :
    a • S ≤ a • T ↔ S ≤ T :=
  set_smul_subset_set_smul_iff


theorem pointwise_smul_le_iff {a : α} {S T : AddSubgroup A} : a • S ≤ T ↔ S ≤ a⁻¹ • T :=
  set_smul_subset_iff


theorem le_pointwise_smul_iff {a : α} {S T : AddSubgroup A} : S ≤ a • T ↔ a⁻¹ • S ≤ T :=
  subset_set_smul_iff


@[simp]
theorem smul_mem_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) (S : AddSubgroup A) (x : A) :
    a • x ∈ a • S ↔ x ∈ S :=
  smul_mem_smul_set_iff₀ ha (S : Set A) x


theorem mem_pointwise_smul_iff_inv_smul_mem₀ {a : α} (ha : a ≠ 0) (S : AddSubgroup A) (x : A) :
    x ∈ a • S ↔ a⁻¹ • x ∈ S :=
  mem_smul_set_iff_inv_smul_mem₀ ha (S : Set A) x


theorem mem_inv_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) (S : AddSubgroup A) (x : A) :
    x ∈ a⁻¹ • S ↔ a • x ∈ S :=
  mem_inv_smul_set_iff₀ ha (S : Set A) x


@[simp]
theorem pointwise_smul_le_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) {S T : AddSubgroup A} :
    a • S ≤ a • T ↔ S ≤ T :=
  set_smul_subset_set_smul_iff₀ ha


theorem pointwise_smul_le_iff₀ {a : α} (ha : a ≠ 0) {S T : AddSubgroup A} :
    a • S ≤ T ↔ S ≤ a⁻¹ • T :=
  set_smul_subset_iff₀ ha


theorem le_pointwise_smul_iff₀ {a : α} (ha : a ≠ 0) {S T : AddSubgroup A} :
    S ≤ a • T ↔ a⁻¹ • S ≤ T :=
  subset_set_smul_iff₀ ha


@[simp] protected lemma zero_smul (s : AddSubgroup M) : (0 : R) • s = ⊥ := by
  /-
    R : Type u_5
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    s : AddSubgroup M
    ⊢ Eq (HSMul.hSMul 0 s) Bot.bot
  -/
  simp [eq_bot_iff_forall, pointwise_smul_def]
  /-
    🎉 no goals
  -/


/-- For additive subgroups `S` and `T` of a ring, the product of `S` and `T` as submonoids
is automatically a subgroup, which we define as the product of `S` and `T` as subgroups. -/
protected def mul : Mul (AddSubgroup R) where
  mul M N :=
  { __ := M.toAddSubmonoid * N.toAddSubmonoid
    neg_mem' := fun h ↦ AddSubmonoid.mul_induction_on h
                          /-
                            α : Type u_1
                            G : Type u_2
                            A : Type u_3
                            S : Type u_4
                            inst✝² : Group G
                            inst✝¹ : AddGroup A
                            s : Set G
                            R : Type u_5
                            inst✝ : NonUnitalNonAssocRing R
                            M N : AddSubgroup R
                            x✝ : R
                            h : Membership.mem __spread✝⁻⁰.carrier x✝
                            m : R
                            hm : Membership.mem M.toAddSubmonoid m
                            n : R
                            hn : Membership.mem N.toAddSubmonoid n
                            ⊢ Membership.mem __spread✝⁻⁰.carrier (Neg.neg (HMul.hMul m n))
                          -/
      (fun m hm n hn ↦ by rw [← neg_mul]; exact AddSubmonoid.mul_mem_mul (M.neg_mem hm) hn)
                                          /-
                                            🎉 no goals
                                          -/
                           /-
                             α : Type u_1
                             G : Type u_2
                             A : Type u_3
                             S : Type u_4
                             inst✝² : Group G
                             inst✝¹ : AddGroup A
                             s : Set G
                             R : Type u_5
                             inst✝ : NonUnitalNonAssocRing R
                             M N : AddSubgroup R
                             x✝ : R
                             h : Membership.mem __spread✝⁻⁰.carrier x✝
                             r₁ r₂ : R
                             h₁ : Membership.mem __spread✝⁻⁰.carrier (Neg.neg r₁)
                             h₂ : Membership.mem __spread✝⁻⁰.carrier (Neg.neg r₂)
                             ⊢ Membership.mem __spread✝⁻⁰.carrier (Neg.neg (HAdd.hAdd r₁ r₂))
                           -/
      fun r₁ r₂ h₁ h₂ ↦ by rw [neg_add]; exact (M.1 * N.1).add_mem h₁ h₂ }
                                         /-
                                           🎉 no goals
                                         -/


theorem mul_toAddSubmonoid (M N : AddSubgroup R) :
    (M * N).toAddSubmonoid = M.toAddSubmonoid * N.toAddSubmonoid := rfl


