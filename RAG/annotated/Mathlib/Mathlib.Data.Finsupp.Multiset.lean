/-- Given `f : α →₀ ℕ`, `f.toMultiset` is the multiset with multiplicities given by the values of
`f` on the elements of `α`. We define this function as an `AddMonoidHom`.

Under the additional assumption of `[DecidableEq α]`, this is available as
`Multiset.toFinsupp : Multiset α ≃+ (α →₀ ℕ)`; the two declarations are separate as this assumption
is only needed for one direction. -/
def toMultiset : (α →₀ ℕ) →+ Multiset α where
  toFun f := Finsupp.sum f fun a n => n • {a}
  -- Porting note: times out if h is not specified
  map_add' _f _g := sum_add_index' (h := fun a n => n • ({a} : Multiset α))
    (fun _ ↦ zero_nsmul _) (fun _ ↦ add_nsmul _)
  map_zero' := sum_zero_index


theorem toMultiset_zero : toMultiset (0 : α →₀ ℕ) = 0 :=
  rfl


theorem toMultiset_add (m n : α →₀ ℕ) : toMultiset (m + n) = toMultiset m + toMultiset n :=
  toMultiset.map_add m n


theorem toMultiset_apply (f : α →₀ ℕ) : toMultiset f = f.sum fun a n => n • {a} :=
  rfl


@[simp]
theorem toMultiset_single (a : α) (n : ℕ) : toMultiset (single a n) = n • {a} := by
  /-
    α : Type u_1
    a : α
    n : Nat
    ⊢ Eq (Finsupp.toMultiset (Finsupp.single a n)) (HSMul.hSMul n (Singleton.singl …
  -/
  rw [toMultiset_apply, sum_single_index]; apply zero_nsmul
                                           /-
                                             🎉 no goals
                                           -/


theorem toMultiset_sum {f : ι → α →₀ ℕ} (s : Finset ι) :
    Finsupp.toMultiset (∑ i ∈ s, f i) = ∑ i ∈ s, Finsupp.toMultiset (f i) :=
  map_sum Finsupp.toMultiset _ _


theorem toMultiset_sum_single (s : Finset ι) (n : ℕ) :
    Finsupp.toMultiset (∑ i ∈ s, single i n) = n • s.val := by
  /-
    ι : Type u_3
    s : Finset ι
    n : Nat
    ⊢ Eq (Finsupp.toMultiset (s.sum fun i => Finsupp.single i n)) (HSMul.hSMul n s …
  -/
  simp_rw [toMultiset_sum, Finsupp.toMultiset_single, Finset.sum_nsmul, sum_multiset_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_toMultiset (f : α →₀ ℕ) : Multiset.card (toMultiset f) = f.sum fun _ => id := by
  /-
    α : Type u_1
    f : Finsupp α Nat
    ⊢ Eq (Finsupp.toMultiset f).card (f.sum fun x => id)
  -/
  simp [toMultiset_apply, map_finsupp_sum, Function.id_def]
  /-
    🎉 no goals
  -/


theorem toMultiset_map (f : α →₀ ℕ) (g : α → β) :
    f.toMultiset.map g = toMultiset (f.mapDomain g) := by
  /-
    α : Type u_1
    β : Type u_2
    f : Finsupp α Nat
    g : α → β
    ⊢ Eq (Multiset.map g (Finsupp.toMultiset f)) (Finsupp.toMultiset (Finsupp.mapD …
  -/
  refine f.induction ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      f : Finsupp α Nat
      g : α → β
      ⊢ Eq (Multiset.map g (Finsupp.toMultiset 0)) (Finsupp.toMultiset (Finsupp.mapD …
    -/
  · rw [toMultiset_zero, Multiset.map_zero, mapDomain_zero, toMultiset_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      f : Finsupp α Nat
      g : α → β
      ⊢ ∀ (a : α) (b : Nat) (f : Finsupp α Nat), Not (Membership.mem f.support a) →  …
    -/
  · intro a n f _ _ ih
    rw [toMultiset_add, Multiset.map_add, ih, mapDomain_add, mapDomain_single,
      toMultiset_single, toMultiset_add, toMultiset_single, ← Multiset.coe_mapAddMonoidHom,
      (Multiset.mapAddMonoidHom g).map_nsmul]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      f✝ : Finsupp α Nat
      g : α → β
      a : α
      n : Nat
      f : Finsupp α Nat
      a✝¹ : Not (Membership.mem f.support a)
      a✝ : Ne n 0
      ih : Eq (Multiset.map g (Finsupp.toMultiset f)) (Finsupp.toMultiset (Finsupp.m …
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul n ((Multiset.mapAddMonoidHom g) (Singleton.single …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem prod_toMultiset [CommMonoid α] (f : α →₀ ℕ) :
    f.toMultiset.prod = f.prod fun a n => a ^ n := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    f : Finsupp α Nat
    ⊢ Eq (Finsupp.toMultiset f).prod (f.prod fun a n => HPow.hPow a n)
  -/
  refine f.induction ?_ ?_
    /-
      case refine_1
      α : Type u_1
      inst✝ : CommMonoid α
      f : Finsupp α Nat
      ⊢ Eq (Finsupp.toMultiset 0).prod (Finsupp.prod 0 fun a n => HPow.hPow a n)
    -/
  · rw [toMultiset_zero, Multiset.prod_zero, Finsupp.prod_zero_index]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : CommMonoid α
      f : Finsupp α Nat
      ⊢ ∀ (a : α) (b : Nat) (f : Finsupp α Nat), Not (Membership.mem f.support a) →  …
    -/
  · intro a n f _ _ ih
    rw [toMultiset_add, Multiset.prod_add, ih, toMultiset_single, Multiset.prod_nsmul,
      Finsupp.prod_add_index' pow_zero pow_add, Finsupp.prod_single_index, Multiset.prod_singleton]
    /-
      case refine_2
      α : Type u_1
      inst✝ : CommMonoid α
      f✝ : Finsupp α Nat
      a : α
      n : Nat
      f : Finsupp α Nat
      a✝¹ : Not (Membership.mem f.support a)
      a✝ : Ne n 0
      ih : Eq (Finsupp.toMultiset f).prod (f.prod fun a n => HPow.hPow a n)
      ⊢ Eq (HPow.hPow a 0) 1
    -/
    exact pow_zero a
    /-
      🎉 no goals
    -/


@[simp]
theorem toFinset_toMultiset [DecidableEq α] (f : α →₀ ℕ) : f.toMultiset.toFinset = f.support := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    f : Finsupp α Nat
    ⊢ Eq (Finsupp.toMultiset f).toFinset f.support
  -/
  refine f.induction ?_ ?_
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      f : Finsupp α Nat
      ⊢ Eq (Finsupp.toMultiset 0).toFinset (Finsupp.support 0)
    -/
  · rw [toMultiset_zero, Multiset.toFinset_zero, support_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      f : Finsupp α Nat
      ⊢ ∀ (a : α) (b : Nat) (f : Finsupp α Nat), Not (Membership.mem f.support a) →  …
    -/
  · intro a n f ha hn ih
    rw [toMultiset_add, Multiset.toFinset_add, ih, toMultiset_single, support_add_eq,
      support_single_ne_zero _ hn, Multiset.toFinset_nsmul _ _ hn, Multiset.toFinset_singleton]
    /-
      case refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      f✝ : Finsupp α Nat
      a : α
      n : Nat
      f : Finsupp α Nat
      ha : Not (Membership.mem f.support a)
      hn : Ne n 0
      ih : Eq (Finsupp.toMultiset f).toFinset f.support
      ⊢ Disjoint (Finsupp.single a n).support f.support
    -/
    refine Disjoint.mono_left support_single_subset ?_
    /-
      case refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      f✝ : Finsupp α Nat
      a : α
      n : Nat
      f : Finsupp α Nat
      ha : Not (Membership.mem f.support a)
      hn : Ne n 0
      ih : Eq (Finsupp.toMultiset f).toFinset f.support
      ⊢ Disjoint (Singleton.singleton a) f.support
    -/
    rwa [Finset.disjoint_singleton_left]
    /-
      🎉 no goals
    -/


@[simp]
theorem count_toMultiset [DecidableEq α] (f : α →₀ ℕ) (a : α) : (toMultiset f).count a = f a :=
  calc
    (toMultiset f).count a = Finsupp.sum f (fun x n => (n • {x} : Multiset α).count a) := by
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        f : Finsupp α Nat
        a : α
        ⊢ Eq (Multiset.count a (Finsupp.toMultiset f)) (f.sum fun x n => Multiset.coun …
      -/
      rw [toMultiset_apply]; exact map_sum (Multiset.countAddMonoidHom a) _ f.support
                             /-
                               🎉 no goals
                             -/
                                                              /-
                                                                α : Type u_1
                                                                inst✝ : DecidableEq α
                                                                f : Finsupp α Nat
                                                                a : α
                                                                ⊢ Eq (f.sum fun x n => Multiset.count a (HSMul.hSMul n (Singleton.singleton x) …
                                                              -/
    _ = f.sum fun x n => n * ({x} : Multiset α).count a := by simp only [Multiset.count_nsmul]
                                                              /-
                                                                🎉 no goals
                                                              -/
    _ = f a * ({a} : Multiset α).count a :=
      sum_eq_single _
                          /-
                            α : Type u_1
                            inst✝ : DecidableEq α
                            f : Finsupp α Nat
                            a a' : α
                            x✝ : Ne (f a') 0
                            H : Ne a' a
                            ⊢ Eq (HMul.hMul (f a') (Multiset.count a (Singleton.singleton a'))) 0
                          -/
        (fun a' _ H => by simp only [Multiset.count_singleton, if_false, H.symm, mul_zero])
                          /-
                            🎉 no goals
                          -/
        (fun _ => zero_mul _)
                  /-
                    α : Type u_1
                    inst✝ : DecidableEq α
                    f : Finsupp α Nat
                    a : α
                    ⊢ Eq (HMul.hMul (f a) (Multiset.count a (Singleton.singleton a))) (f a)
                  -/
    _ = f a := by rw [Multiset.count_singleton_self, mul_one]
                  /-
                    🎉 no goals
                  -/


theorem toMultiset_sup [DecidableEq α] (f g : α →₀ ℕ) :
    toMultiset (f ⊔ g) = toMultiset f ∪ toMultiset g := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    f g : Finsupp α Nat
    ⊢ Eq (Finsupp.toMultiset (Max.max f g)) (Union.union (Finsupp.toMultiset f) (F …
  -/
  ext
  /-
    case a
    α : Type u_1
    inst✝ : DecidableEq α
    f g : Finsupp α Nat
    a✝ : α
    ⊢ Eq (Multiset.count a✝ (Finsupp.toMultiset (Max.max f g))) (Multiset.count a✝ …
  -/
  simp_rw [Multiset.count_union, Finsupp.count_toMultiset, Finsupp.sup_apply]
  /-
    🎉 no goals
  -/


theorem toMultiset_inf [DecidableEq α] (f g : α →₀ ℕ) :
    toMultiset (f ⊓ g) = toMultiset f ∩ toMultiset g := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    f g : Finsupp α Nat
    ⊢ Eq (Finsupp.toMultiset (Min.min f g)) (Inter.inter (Finsupp.toMultiset f) (F …
  -/
  ext
  /-
    case a
    α : Type u_1
    inst✝ : DecidableEq α
    f g : Finsupp α Nat
    a✝ : α
    ⊢ Eq (Multiset.count a✝ (Finsupp.toMultiset (Min.min f g))) (Multiset.count a✝ …
  -/
  simp_rw [Multiset.count_inter, Finsupp.count_toMultiset, Finsupp.inf_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_toMultiset (f : α →₀ ℕ) (i : α) : i ∈ toMultiset f ↔ i ∈ f.support := by
  classical
  rw [← Multiset.count_ne_zero, Finsupp.count_toMultiset, Finsupp.mem_support_iff]


/-- Given a multiset `s`, `s.toFinsupp` returns the finitely supported function on `ℕ` given by
the multiplicities of the elements of `s`. -/
@[simps symm_apply]
def toFinsupp : Multiset α ≃+ (α →₀ ℕ) where
                                                          /-
                                                            α : Type u_1
                                                            β : Type u_2
                                                            ι : Type u_3
                                                            inst✝ : DecidableEq α
                                                            s : Multiset α
                                                            a : α
                                                            ⊢ Iff (Membership.mem s.toFinset a) (Ne ((fun a => Multiset.count a s) a) 0)
                                                          -/
  toFun s := ⟨s.toFinset, fun a => s.count a, fun a => by simp⟩
                                                          /-
                                                            🎉 no goals
                                                          -/
  invFun f := Finsupp.toMultiset f
  map_add' _ _ := Finsupp.ext fun _ => count_add _ _ _
  right_inv f :=
    Finsupp.ext fun a => by
      simp only [Finsupp.toMultiset_apply, Finsupp.sum, Multiset.count_sum',
        Multiset.count_singleton, mul_boole, Finsupp.coe_mk, Finsupp.mem_support_iff,
        Multiset.count_nsmul, Finset.sum_ite_eq, ite_not, ite_eq_right_iff]
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        inst✝ : DecidableEq α
        f : Finsupp α Nat
        a : α
        ⊢ Eq (f a) 0 → Eq 0 (f a)
      -/
      exact Eq.symm
      /-
        🎉 no goals
      -/
  left_inv s := by simp only [Finsupp.toMultiset_apply, Finsupp.sum, Finsupp.coe_mk,
    Multiset.toFinset_sum_count_nsmul_eq]


@[simp]
theorem toFinsupp_support (s : Multiset α) : s.toFinsupp.support = s.toFinset := rfl


@[simp]
theorem toFinsupp_apply (s : Multiset α) (a : α) : toFinsupp s a = s.count a := rfl


theorem toFinsupp_zero : toFinsupp (0 : Multiset α) = 0 := _root_.map_zero _


theorem toFinsupp_add (s t : Multiset α) : toFinsupp (s + t) = toFinsupp s + toFinsupp t :=
  _root_.map_add toFinsupp s t


@[simp]
theorem toFinsupp_singleton (a : α) : toFinsupp ({a} : Multiset α) = Finsupp.single a 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    ⊢ Eq (Multiset.toFinsupp (Singleton.singleton a)) (Finsupp.single a 1)
  -/
  ext; rw [toFinsupp_apply, count_singleton, Finsupp.single_eq_pi_single, Pi.single_apply]
       /-
         🎉 no goals
       -/


@[simp]
theorem toFinsupp_toMultiset (s : Multiset α) : Finsupp.toMultiset (toFinsupp s) = s :=
  Multiset.toFinsupp.symm_apply_apply s


theorem toFinsupp_eq_iff {s : Multiset α} {f : α →₀ ℕ} :
    toFinsupp s = f ↔ s = Finsupp.toMultiset f :=
  Multiset.toFinsupp.apply_eq_iff_symm_apply


theorem toFinsupp_union (s t : Multiset α) : toFinsupp (s ∪ t) = toFinsupp s ⊔ toFinsupp t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Eq (Multiset.toFinsupp (Union.union s t)) (Max.max (Multiset.toFinsupp s) (M …
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    a✝ : α
    ⊢ Eq ((Multiset.toFinsupp (Union.union s t)) a✝) ((Max.max (Multiset.toFinsupp …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem toFinsupp_inter (s t : Multiset α) : toFinsupp (s ∩ t) = toFinsupp s ⊓ toFinsupp t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Eq (Multiset.toFinsupp (Inter.inter s t)) (Min.min (Multiset.toFinsupp s) (M …
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    a✝ : α
    ⊢ Eq ((Multiset.toFinsupp (Inter.inter s t)) a✝) ((Min.min (Multiset.toFinsupp …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinsupp_sum_eq (s : Multiset α) : s.toFinsupp.sum (fun _ ↦ id) = Multiset.card s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    ⊢ Eq ((Multiset.toFinsupp s).sum fun x => id) s.card
  -/
  rw [← Finsupp.card_toMultiset, toFinsupp_toMultiset]
  /-
    🎉 no goals
  -/


@[simp]
theorem Finsupp.toMultiset_toFinsupp [DecidableEq α] (f : α →₀ ℕ) :
    Multiset.toFinsupp (Finsupp.toMultiset f) = f :=
  Multiset.toFinsupp.apply_symm_apply _


theorem Finsupp.toMultiset_eq_iff [DecidableEq α] {f : α →₀ ℕ} {s : Multiset α} :
    Finsupp.toMultiset f = s ↔ f = Multiset.toFinsupp s :=
  Multiset.toFinsupp.symm_apply_eq


/-- `Finsupp.toMultiset` as an order isomorphism. -/
def orderIsoMultiset [DecidableEq ι] : (ι →₀ ℕ) ≃o Multiset ι where
  toEquiv := Multiset.toFinsupp.symm.toEquiv
                           /-
                             α : Type u_1
                             β : Type u_2
                             ι : Type u_3
                             inst✝ : DecidableEq ι
                             f g : Finsupp ι Nat
                             ⊢ Iff (LE.le (Multiset.toFinsupp.symm.toEquiv f) (Multiset.toFinsupp.symm.toEq …
                           -/
  map_rel_iff' {f g} := by simp [le_def, Multiset.le_iff_count]
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem coe_orderIsoMultiset [DecidableEq ι] : ⇑(@orderIsoMultiset ι _) = toMultiset :=
  rfl


@[simp]
theorem coe_orderIsoMultiset_symm [DecidableEq ι] :
    ⇑(@orderIsoMultiset ι).symm = Multiset.toFinsupp :=
  rfl


theorem toMultiset_strictMono : StrictMono (@toMultiset ι) := by
  /-
    ι : Type u_3
    ⊢ StrictMono ⇑Finsupp.toMultiset
  -/
  classical exact (@orderIsoMultiset ι _).strictMono
  /-
    🎉 no goals
  -/


theorem sum_id_lt_of_lt (m n : ι →₀ ℕ) (h : m < n) : (m.sum fun _ => id) < n.sum fun _ => id := by
  /-
    ι : Type u_3
    m n : Finsupp ι Nat
    h : LT.lt m n
    ⊢ LT.lt (m.sum fun x => id) (n.sum fun x => id)
  -/
  rw [← card_toMultiset, ← card_toMultiset]
  /-
    ι : Type u_3
    m n : Finsupp ι Nat
    h : LT.lt m n
    ⊢ LT.lt (Finsupp.toMultiset m).card (Finsupp.toMultiset n).card
  -/
  apply Multiset.card_lt_card
  /-
    case h
    ι : Type u_3
    m n : Finsupp ι Nat
    h : LT.lt m n
    ⊢ LT.lt (Finsupp.toMultiset m) (Finsupp.toMultiset n)
  -/
  exact toMultiset_strictMono h
  /-
    🎉 no goals
  -/


/-- The order on `ι →₀ ℕ` is well-founded. -/
theorem lt_wf : WellFounded (@LT.lt (ι →₀ ℕ) _) :=
  Subrelation.wf (sum_id_lt_of_lt _ _) <| InvImage.wf _ Nat.lt_wfRel.2

-- TODO: generalize to `[WellFoundedRelation α] → WellFoundedRelation (ι →₀ α)`

instance : WellFoundedRelation (ι →₀ ℕ) where
  rel := (· < ·)
  wf := lt_wf _


theorem Multiset.toFinsupp_strictMono [DecidableEq ι] : StrictMono (@Multiset.toFinsupp ι _) :=
  (@Finsupp.orderIsoMultiset ι).symm.strictMono


/-- The `n`th symmetric power of a type `α` is naturally equivalent to the subtype of
finitely-supported maps `α →₀ ℕ` with total mass `n`.

See also `Sym.equivNatSumOfFintype` when `α` is finite. -/
def equivNatSum :
    Sym α n ≃ {P : α →₀ ℕ // P.sum (fun _ ↦ id) = n} :=
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  ι : Type u_3
                                                  inst✝ : DecidableEq α
                                                  n : Nat
                                                  ⊢ ∀ (a : Multiset α), Iff (Eq a.card n) (Eq ((Multiset.toFinsupp.toEquiv a).su …
                                                -/
  Multiset.toFinsupp.toEquiv.subtypeEquiv <| by simp
                                                /-
                                                  🎉 no goals
                                                -/


@[simp] lemma coe_equivNatSum_apply_apply (s : Sym α n) (a : α) :
    (equivNatSum α n s : α →₀ ℕ) a = (s : Multiset α).count a :=
  rfl


@[simp] lemma coe_equivNatSum_symm_apply (P : {P : α →₀ ℕ // P.sum (fun _ ↦ id) = n}) :
    ((equivNatSum α n).symm P : Multiset α) = Finsupp.toMultiset P :=
  rfl


/-- The `n`th symmetric power of a finite type `α` is naturally equivalent to the subtype of maps
`α → ℕ` with total mass `n`.

See also `Sym.equivNatSum` when `α` is not necessarily finite. -/
noncomputable def equivNatSumOfFintype [Fintype α] :
    Sym α n ≃ {P : α → ℕ // ∑ i, P i = n} :=
                                                                         /-
                                                                           α : Type u_1
                                                                           β : Type u_2
                                                                           ι : Type u_3
                                                                           inst✝¹ : DecidableEq α
                                                                           n : Nat
                                                                           inst✝ : Fintype α
                                                                           ⊢ ∀ (a : Finsupp α Nat), Iff (Eq (a.sum fun x => id) n) (Eq (Finset.univ.sum f …
                                                                         -/
  (equivNatSum α n).trans <| Finsupp.equivFunOnFinite.subtypeEquiv <| by simp [Finsupp.sum_fintype]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp] lemma coe_equivNatSumOfFintype_apply_apply [Fintype α] (s : Sym α n) (a : α) :
    (equivNatSumOfFintype α n s : α → ℕ) a = (s : Multiset α).count a :=
  rfl


@[simp] lemma coe_equivNatSumOfFintype_symm_apply [Fintype α] (P : {P : α → ℕ // ∑ i, P i = n}) :
    ((equivNatSumOfFintype α n).symm P : Multiset α) = ∑ a, ((P : α → ℕ) a) • {a} := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    n : Nat
    inst✝ : Fintype α
    P : Subtype fun P => Eq (Finset.univ.sum fun i => P i) n
    ⊢ Eq (↑((Sym.equivNatSumOfFintype α n).symm P)) (Finset.univ.sum fun a => HSMu …
  -/
  obtain ⟨P, hP⟩ := P
  /-
    case mk
    α : Type u_1
    inst✝¹ : DecidableEq α
    n : Nat
    inst✝ : Fintype α
    P : α → Nat
    hP : Eq (Finset.univ.sum fun i => P i) n
    ⊢ Eq (↑((Sym.equivNatSumOfFintype α n).symm ⟨P, hP⟩)) (Finset.univ.sum fun a = …
  -/
  change Finsupp.toMultiset (Finsupp.equivFunOnFinite.symm P) = Multiset.sum _
  /-
    case mk
    α : Type u_1
    inst✝¹ : DecidableEq α
    n : Nat
    inst✝ : Fintype α
    P : α → Nat
    hP : Eq (Finset.univ.sum fun i => P i) n
    ⊢ Eq (Finsupp.toMultiset (Finsupp.equivFunOnFinite.symm P)) (Multiset.map (fun …
  -/
  ext a
  /-
    case mk.a
    α : Type u_1
    inst✝¹ : DecidableEq α
    n : Nat
    inst✝ : Fintype α
    P : α → Nat
    hP : Eq (Finset.univ.sum fun i => P i) n
    a : α
    ⊢ Eq (Multiset.count a (Finsupp.toMultiset (Finsupp.equivFunOnFinite.symm P))) …
  -/
  rw [Multiset.count_sum]
  /-
    case mk.a
    α : Type u_1
    inst✝¹ : DecidableEq α
    n : Nat
    inst✝ : Fintype α
    P : α → Nat
    hP : Eq (Finset.univ.sum fun i => P i) n
    a : α
    ⊢ Eq (Multiset.count a (Finsupp.toMultiset (Finsupp.equivFunOnFinite.symm P))) …
  -/
  simp [Multiset.count_singleton]
  /-
    🎉 no goals
  -/


