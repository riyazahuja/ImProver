/-- `mulSupport` of a function is the set of points `x` such that `f x ≠ 1`. -/
@[to_additive "`support` of a function is the set of points `x` such that `f x ≠ 0`."]
def mulSupport (f : α → M) : Set α := {x | f x ≠ 1}


@[to_additive]
theorem mulSupport_eq_preimage (f : α → M) : mulSupport f = f ⁻¹' {1}ᶜ :=
  rfl


@[to_additive]
theorem nmem_mulSupport {f : α → M} {x : α} : x ∉ mulSupport f ↔ f x = 1 :=
  not_not


@[to_additive]
theorem compl_mulSupport {f : α → M} : (mulSupport f)ᶜ = { x | f x = 1 } :=
  ext fun _ => nmem_mulSupport


@[to_additive (attr := simp)]
theorem mem_mulSupport {f : α → M} {x : α} : x ∈ mulSupport f ↔ f x ≠ 1 :=
  Iff.rfl


@[to_additive (attr := simp)]
theorem mulSupport_subset_iff {f : α → M} {s : Set α} : mulSupport f ⊆ s ↔ ∀ x, f x ≠ 1 → x ∈ s :=
  Iff.rfl


@[to_additive]
theorem mulSupport_subset_iff' {f : α → M} {s : Set α} :
    mulSupport f ⊆ s ↔ ∀ x ∉ s, f x = 1 :=
  forall_congr' fun _ => not_imp_comm


@[to_additive]
theorem mulSupport_eq_iff {f : α → M} {s : Set α} :
    mulSupport f = s ↔ (∀ x, x ∈ s → f x ≠ 1) ∧ ∀ x, x ∉ s → f x = 1 := by
  simp +contextual only [Set.ext_iff, mem_mulSupport, ne_eq, iff_def,
    not_imp_comm, and_comm, forall_and]


@[to_additive]
theorem ext_iff_mulSupport {f g : α → M} :
    f = g ↔ f.mulSupport = g.mulSupport ∧ ∀ x ∈ f.mulSupport, f x = g x :=
  ⟨fun h ↦ h ▸ ⟨rfl, fun _ _ ↦ rfl⟩, fun ⟨h₁, h₂⟩ ↦ funext fun x ↦ by
    if hx : x ∈ f.mulSupport then exact h₂ x hx
    else rw [nmem_mulSupport.1 hx, nmem_mulSupport.1 (mt (Set.ext_iff.1 h₁ x).2 hx)]⟩


@[to_additive]
theorem mulSupport_update_of_ne_one [DecidableEq α] (f : α → M) (x : α) {y : M} (hy : y ≠ 1) :
    mulSupport (update f x y) = insert x (mulSupport f) := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : One M
    inst✝ : DecidableEq α
    f : α → M
    x : α
    y : M
    hy : Ne y 1
    ⊢ Eq (Function.mulSupport (Function.update f x y)) (Insert.insert x (Function. …
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  ext a; rcases eq_or_ne a x with rfl | hne <;> simp [*]
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive]
theorem mulSupport_update_one [DecidableEq α] (f : α → M) (x : α) :
    mulSupport (update f x 1) = mulSupport f \ {x} := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝¹ : One M
    inst✝ : DecidableEq α
    f : α → M
    x : α
    ⊢ Eq (Function.mulSupport (Function.update f x 1)) (SDiff.sdiff (Function.mulS …
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  ext a; rcases eq_or_ne a x with rfl | hne <;> simp [*]
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive]
theorem mulSupport_update_eq_ite [DecidableEq α] [DecidableEq M] (f : α → M) (x : α) (y : M) :
    mulSupport (update f x y) = if y = 1 then mulSupport f \ {x} else insert x (mulSupport f) := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝² : One M
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq M
    f : α → M
    x : α
    y : M
    ⊢ Eq (Function.mulSupport (Function.update f x y)) (ite (Eq y 1) (SDiff.sdiff  …
  -/
                                        /-
                                          🎉 no goals
                                        -/
  rcases eq_or_ne y 1 with rfl | hy <;> simp [mulSupport_update_one, mulSupport_update_of_ne_one, *]
                                        /-
                                          🎉 no goals
                                        -/


@[to_additive]
theorem mulSupport_extend_one_subset {f : α → M'} {g : α → N} :
    mulSupport (f.extend g 1) ⊆ f '' mulSupport g :=
  mulSupport_subset_iff'.mpr fun x hfg ↦ by
    /-
      α : Type u_1
      M' : Type u_6
      N : Type u_7
      inst✝ : One N
      f : α → M'
      g : α → N
      x : M'
      hfg : Not (Membership.mem (Set.image f (Function.mulSupport g)) x)
      ⊢ Eq (Function.extend f g 1 x) 1
    -/
    by_cases hf : ∃ a, f a = x
      /-
        case pos
        α : Type u_1
        M' : Type u_6
        N : Type u_7
        inst✝ : One N
        f : α → M'
        g : α → N
        x : M'
        hfg : Not (Membership.mem (Set.image f (Function.mulSupport g)) x)
        hf : Exists fun a => Eq (f a) x
        ⊢ Eq (Function.extend f g 1 x) 1
      -/
    · rw [extend, dif_pos hf, ← nmem_mulSupport]
      /-
        case pos
        α : Type u_1
        M' : Type u_6
        N : Type u_7
        inst✝ : One N
        f : α → M'
        g : α → N
        x : M'
        hfg : Not (Membership.mem (Set.image f (Function.mulSupport g)) x)
        hf : Exists fun a => Eq (f a) x
        ⊢ Not (Membership.mem (Function.mulSupport g) (Classical.choose hf))
      -/
      rw [← Classical.choose_spec hf] at hfg
      /-
        case pos
        α : Type u_1
        M' : Type u_6
        N : Type u_7
        inst✝ : One N
        f : α → M'
        g : α → N
        x : M'
        hf : Exists fun a => Eq (f a) x
        hfg : Not (Membership.mem (Set.image f (Function.mulSupport g)) (f (Classical. …
        ⊢ Not (Membership.mem (Function.mulSupport g) (Classical.choose hf))
      -/
      exact fun hg ↦ hfg ⟨_, hg, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        M' : Type u_6
        N : Type u_7
        inst✝ : One N
        f : α → M'
        g : α → N
        x : M'
        hfg : Not (Membership.mem (Set.image f (Function.mulSupport g)) x)
        hf : Not (Exists fun a => Eq (f a) x)
        ⊢ Eq (Function.extend f g 1 x) 1
      -/
    · rw [extend_apply' _ _ _ hf]; rfl
                                   /-
                                     🎉 no goals
                                   -/


@[to_additive]
theorem mulSupport_extend_one {f : α → M'} {g : α → N} (hf : f.Injective) :
    mulSupport (f.extend g 1) = f '' mulSupport g :=
  mulSupport_extend_one_subset.antisymm <| by
    /-
      α : Type u_1
      M' : Type u_6
      N : Type u_7
      inst✝ : One N
      f : α → M'
      g : α → N
      hf : Function.Injective f
      ⊢ HasSubset.Subset (Set.image f (Function.mulSupport g)) (Function.mulSupport  …
    -/
    rintro _ ⟨x, hx, rfl⟩; rwa [mem_mulSupport, hf.extend_apply]
                           /-
                             🎉 no goals
                           -/


@[to_additive]
theorem mulSupport_disjoint_iff {f : α → M} {s : Set α} :
    Disjoint (mulSupport f) s ↔ EqOn f 1 s := by
  simp_rw [← subset_compl_iff_disjoint_right, mulSupport_subset_iff', not_mem_compl_iff, EqOn,
    Pi.one_apply]


@[to_additive]
theorem disjoint_mulSupport_iff {f : α → M} {s : Set α} :
    Disjoint s (mulSupport f) ↔ EqOn f 1 s := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : One M
    f : α → M
    s : Set α
    ⊢ Iff (Disjoint s (Function.mulSupport f)) (Set.EqOn f 1 s)
  -/
  rw [disjoint_comm, mulSupport_disjoint_iff]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mulSupport_eq_empty_iff {f : α → M} : mulSupport f = ∅ ↔ f = 1 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : One M
    f : α → M
    ⊢ Iff (Eq (Function.mulSupport f) EmptyCollection.emptyCollection) (Eq f 1)
  -/
  rw [← subset_empty_iff, mulSupport_subset_iff', funext_iff]
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : One M
    f : α → M
    ⊢ Iff (∀ (x : α), Not (Membership.mem EmptyCollection.emptyCollection x) → Eq  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mulSupport_nonempty_iff {f : α → M} : (mulSupport f).Nonempty ↔ f ≠ 1 := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : One M
    f : α → M
    ⊢ Iff (Function.mulSupport f).Nonempty (Ne f 1)
  -/
  rw [nonempty_iff_ne_empty, Ne, mulSupport_eq_empty_iff]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem range_subset_insert_image_mulSupport (f : α → M) :
    range f ⊆ insert 1 (f '' mulSupport f) := by
  simpa only [range_subset_iff, mem_insert_iff, or_iff_not_imp_left] using
    fun x (hx : x ∈ mulSupport f) => mem_image_of_mem f hx


@[to_additive]
lemma range_eq_image_or_of_mulSupport_subset {f : α → M} {k : Set α} (h : mulSupport f ⊆ k) :
    range f = f '' k ∨ range f = insert 1 (f '' k) := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : One M
    f : α → M
    k : Set α
    h : HasSubset.Subset (Function.mulSupport f) k
    ⊢ Or (Eq (Set.range f) (Set.image f k)) (Eq (Set.range f) (Insert.insert 1 (Se …
  -/
  apply (wcovBy_insert _ _).eq_or_eq (image_subset_range _ _)
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : One M
    f : α → M
    k : Set α
    h : HasSubset.Subset (Function.mulSupport f) k
    ⊢ LE.le (Set.range f) (Insert.insert 1 (Set.image f k))
  -/
  exact (range_subset_insert_image_mulSupport f).trans (insert_subset_insert (image_subset f h))
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mulSupport_one' : mulSupport (1 : α → M) = ∅ :=
  mulSupport_eq_empty_iff.2 rfl


@[to_additive (attr := simp)]
theorem mulSupport_one : (mulSupport fun _ : α => (1 : M)) = ∅ :=
  mulSupport_one'


@[to_additive]
theorem mulSupport_const {c : M} (hc : c ≠ 1) : (mulSupport fun _ : α => c) = Set.univ := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : One M
    c : M
    hc : Ne c 1
    ⊢ Eq (Function.mulSupport fun x => c) Set.univ
  -/
  ext x
  /-
    case h
    α : Type u_1
    M : Type u_5
    inst✝ : One M
    c : M
    hc : Ne c 1
    x : α
    ⊢ Iff (Membership.mem (Function.mulSupport fun x => c) x) (Membership.mem Set. …
  -/
  simp [hc]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulSupport_binop_subset (op : M → N → P) (op1 : op 1 1 = 1) (f : α → M) (g : α → N) :
    (mulSupport fun x => op (f x) (g x)) ⊆ mulSupport f ∪ mulSupport g := fun x hx =>
                                      /-
                                        α : Type u_1
                                        M : Type u_5
                                        N : Type u_7
                                        P : Type u_8
                                        inst✝² : One M
                                        inst✝¹ : One N
                                        inst✝ : One P
                                        op : M → N → P
                                        op1 : Eq (op 1 1) 1
                                        f : α → M
                                        g : α → N
                                        x : α
                                        hx : Membership.mem (Function.mulSupport fun x => op (f x) (g x)) x
                                        hf : Eq (f x) 1
                                        hg : Eq (g x) 1
                                        ⊢ Eq ((fun x => op (f x) (g x)) x) 1
                                      -/
  not_or_of_imp fun hf hg => hx <| by simp only [hf, hg, op1]
                                      /-
                                        🎉 no goals
                                      -/


@[to_additive]
theorem mulSupport_comp_subset {g : M → N} (hg : g 1 = 1) (f : α → M) :
                                                                 /-
                                                                   α : Type u_1
                                                                   M : Type u_5
                                                                   N : Type u_7
                                                                   inst✝¹ : One M
                                                                   inst✝ : One N
                                                                   g : M → N
                                                                   hg : Eq (g 1) 1
                                                                   f : α → M
                                                                   x : α
                                                                   h : Eq (f x) 1
                                                                   ⊢ Eq (Function.comp g f x) 1
                                                                 -/
    mulSupport (g ∘ f) ⊆ mulSupport f := fun x => mt fun h => by simp only [(· ∘ ·), *]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[to_additive]
theorem mulSupport_subset_comp {g : M → N} (hg : ∀ {x}, g x = 1 → x = 1) (f : α → M) :
    mulSupport f ⊆ mulSupport (g ∘ f) := fun _ => mt hg


@[to_additive]
theorem mulSupport_comp_eq (g : M → N) (hg : ∀ {x}, g x = 1 ↔ x = 1) (f : α → M) :
    mulSupport (g ∘ f) = mulSupport f :=
  Set.ext fun _ => not_congr hg


@[to_additive]
theorem mulSupport_comp_eq_of_range_subset {g : M → N} {f : α → M}
    (hg : ∀ {x}, x ∈ range f → (g x = 1 ↔ x = 1)) :
    mulSupport (g ∘ f) = mulSupport f :=
                                  /-
                                    α : Type u_1
                                    M : Type u_5
                                    N : Type u_7
                                    inst✝¹ : One M
                                    inst✝ : One N
                                    g : M → N
                                    f : α → M
                                    hg : ∀ {x : M}, Membership.mem (Set.range f) x → Iff (Eq (g x) 1) (Eq x 1)
                                    x : α
                                    ⊢ Iff (Eq (Function.comp g f x) 1) (Eq (f x) 1)
                                  -/
  Set.ext fun x ↦ not_congr <| by rw [Function.comp, hg (mem_range_self x)]
                                  /-
                                    🎉 no goals
                                  -/


@[to_additive]
theorem mulSupport_comp_eq_preimage (g : β → M) (f : α → β) :
    mulSupport (g ∘ f) = f ⁻¹' mulSupport g :=
  rfl


@[to_additive support_prod_mk]
theorem mulSupport_prod_mk (f : α → M) (g : α → N) :
    (mulSupport fun x => (f x, g x)) = mulSupport f ∪ mulSupport g :=
  Set.ext fun x => by
    /-
      α : Type u_1
      M : Type u_5
      N : Type u_7
      inst✝¹ : One M
      inst✝ : One N
      f : α → M
      g : α → N
      x : α
      ⊢ Iff (Membership.mem (Function.mulSupport fun x => { fst := f x, snd := g x } …
    -/
    simp only [mulSupport, not_and_or, mem_union, mem_setOf_eq, Prod.mk_eq_one, Ne]
    /-
      🎉 no goals
    -/


@[to_additive support_prod_mk']
theorem mulSupport_prod_mk' (f : α → M × N) :
    mulSupport f = (mulSupport fun x => (f x).1) ∪ mulSupport fun x => (f x).2 := by
  /-
    α : Type u_1
    M : Type u_5
    N : Type u_7
    inst✝¹ : One M
    inst✝ : One N
    f : α → Prod M N
    ⊢ Eq (Function.mulSupport f) (Union.union (Function.mulSupport fun x => (f x). …
  -/
  simp only [← mulSupport_prod_mk]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulSupport_along_fiber_subset (f : α × β → M) (a : α) :
    (mulSupport fun b => f (a, b)) ⊆ (mulSupport f).image Prod.snd :=
                          /-
                            α : Type u_1
                            β : Type u_2
                            M : Type u_5
                            inst✝ : One M
                            f : Prod α β → M
                            a : α
                            x : β
                            hx : Membership.mem (Function.mulSupport fun b => f { fst := a, snd := b }) x
                            ⊢ And (Membership.mem (Function.mulSupport f) { fst := a, snd := x }) (Eq { fs …
                          -/
  fun x hx => ⟨(a, x), by simpa using hx⟩
                          /-
                            🎉 no goals
                          -/


@[to_additive]
theorem mulSupport_curry (f : α × β → M) :
    (mulSupport f.curry) = (mulSupport f).image Prod.fst := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_5
    inst✝ : One M
    f : Prod α β → M
    ⊢ Eq (Function.mulSupport (Function.curry f)) (Set.image Prod.fst (Function.mu …
  -/
  simp [mulSupport, funext_iff, image]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulSupport_curry' (f : α × β → M) :
    (mulSupport fun a b ↦ f (a, b)) = (mulSupport f).image Prod.fst :=
  mulSupport_curry f


@[to_additive]
theorem mulSupport_mul [MulOneClass M] (f g : α → M) :
    (mulSupport fun x => f x * g x) ⊆ mulSupport f ∪ mulSupport g :=
  mulSupport_binop_subset (· * ·) (one_mul _) f g


@[to_additive]
theorem mulSupport_pow [Monoid M] (f : α → M) (n : ℕ) :
    (mulSupport fun x => f x ^ n) ⊆ mulSupport f := by
  induction n with
  | zero => simp [pow_zero, mulSupport_one]
  | succ n hfn =>
    simpa only [pow_succ'] using (mulSupport_mul f _).trans (union_subset Subset.rfl hfn)


@[to_additive (attr := simp)]
theorem mulSupport_inv : (mulSupport fun x => (f x)⁻¹) = mulSupport f :=
  ext fun _ => inv_ne_one


@[to_additive (attr := simp)]
theorem mulSupport_inv' : mulSupport f⁻¹ = mulSupport f :=
  mulSupport_inv f


@[to_additive]
theorem mulSupport_mul_inv : (mulSupport fun x => f x * (g x)⁻¹) ⊆ mulSupport f ∪ mulSupport g :=
                                                   /-
                                                     α : Type u_1
                                                     G : Type u_9
                                                     inst✝ : DivisionMonoid G
                                                     f g : α → G
                                                     ⊢ Eq ((fun a b => HMul.hMul a (Inv.inv b)) 1 1) 1
                                                   -/
  mulSupport_binop_subset (fun a b => a * b⁻¹) (by simp) f g
                                                   /-
                                                     🎉 no goals
                                                   -/


@[to_additive]
theorem mulSupport_div : (mulSupport fun x => f x / g x) ⊆ mulSupport f ∪ mulSupport g :=
  mulSupport_binop_subset (· / ·) one_div_one f g


@[to_additive]
theorem image_inter_mulSupport_eq {s : Set β} {g : β → α} :
    g '' s ∩ mulSupport f = g '' (s ∩ mulSupport (f ∘ g)) := by
  /-
    α : Type u_1
    β : Type u_2
    M : Type u_3
    inst✝ : One M
    f : α → M
    s : Set β
    g : β → α
    ⊢ Eq (Inter.inter (Set.image g s) (Function.mulSupport f)) (Set.image g (Inter …
  -/
  rw [mulSupport_comp_eq_preimage f g, image_inter_preimage]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulSupport_mulSingle_subset : mulSupport (mulSingle a b) ⊆ {a} := fun _ hx =>
  by_contra fun hx' => hx <| mulSingle_eq_of_ne hx' _


@[to_additive]
                                                                              /-
                                                                                A : Type u_1
                                                                                B : Type u_2
                                                                                inst✝¹ : DecidableEq A
                                                                                inst✝ : One B
                                                                                a : A
                                                                                ⊢ Eq (Function.mulSupport (Pi.mulSingle a 1)) EmptyCollection.emptyCollection
                                                                              -/
theorem mulSupport_mulSingle_one : mulSupport (mulSingle a (1 : B)) = ∅ := by simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[to_additive (attr := simp)]
theorem mulSupport_mulSingle_of_ne (h : b ≠ 1) : mulSupport (mulSingle a b) = {a} :=
  mulSupport_mulSingle_subset.antisymm fun x (hx : x = a) => by
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : DecidableEq A
      inst✝ : One B
      a : A
      b : B
      h : Ne b 1
      x : A
      hx : Eq x a
      ⊢ Membership.mem (Function.mulSupport (Pi.mulSingle a b)) x
    -/
    rwa [mem_mulSupport, hx, mulSingle_eq_same]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem mulSupport_mulSingle [DecidableEq B] :
                                                                /-
                                                                  A : Type u_1
                                                                  B : Type u_2
                                                                  inst✝² : DecidableEq A
                                                                  inst✝¹ : One B
                                                                  a : A
                                                                  b : B
                                                                  inst✝ : DecidableEq B
                                                                  ⊢ Eq (Function.mulSupport (Pi.mulSingle a b)) (ite (Eq b 1) EmptyCollection.em …
                                                                -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
    mulSupport (mulSingle a b) = if b = 1 then ∅ else {a} := by split_ifs with h <;> simp [h]
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[to_additive]
theorem mulSupport_mulSingle_disjoint {b' : B} (hb : b ≠ 1) (hb' : b' ≠ 1) {i j : A} :
    Disjoint (mulSupport (mulSingle i b)) (mulSupport (mulSingle j b')) ↔ i ≠ j := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝¹ : DecidableEq A
    inst✝ : One B
    b b' : B
    hb : Ne b 1
    hb' : Ne b' 1
    i j : A
    ⊢ Iff (Disjoint (Function.mulSupport (Pi.mulSingle i b)) (Function.mulSupport  …
  -/
  rw [mulSupport_mulSingle_of_ne hb, mulSupport_mulSingle_of_ne hb', disjoint_singleton]
  /-
    🎉 no goals
  -/


