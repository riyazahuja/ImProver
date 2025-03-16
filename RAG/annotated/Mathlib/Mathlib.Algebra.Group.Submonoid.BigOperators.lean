@[to_additive (attr := norm_cast, simp)]
theorem coe_list_prod (l : List S) : (l.prod : M) = (l.map (↑)).prod :=
  map_list_prod (SubmonoidClass.subtype S : _ →* M) l


@[to_additive (attr := norm_cast, simp)]
theorem coe_multiset_prod {M} [CommMonoid M] [SetLike B M] [SubmonoidClass B M] (m : Multiset S) :
    (m.prod : M) = (m.map (↑)).prod :=
  (SubmonoidClass.subtype S : _ →* M).map_multiset_prod m


@[to_additive (attr := norm_cast, simp)]
theorem coe_finset_prod {ι M} [CommMonoid M] [SetLike B M] [SubmonoidClass B M] (f : ι → S)
    (s : Finset ι) : ↑(∏ i ∈ s, f i) = (∏ i ∈ s, f i : M) :=
  map_prod (SubmonoidClass.subtype S) f s


/-- Product of a list of elements in a submonoid is in the submonoid. -/
@[to_additive "Sum of a list of elements in an `AddSubmonoid` is in the `AddSubmonoid`."]
theorem list_prod_mem {l : List M} (hl : ∀ x ∈ l, x ∈ S) : l.prod ∈ S := by
  /-
    M : Type u_1
    B : Type u_3
    inst✝² : Monoid M
    inst✝¹ : SetLike B M
    inst✝ : SubmonoidClass B M
    S : B
    l : List M
    hl : ∀ (x : M), Membership.mem l x → Membership.mem S x
    ⊢ Membership.mem S l.prod
  -/
  lift l to List S using hl
  /-
    case intro
    M : Type u_1
    B : Type u_3
    inst✝² : Monoid M
    inst✝¹ : SetLike B M
    inst✝ : SubmonoidClass B M
    S : B
    l : List (Subtype fun x => Membership.mem S x)
    ⊢ Membership.mem S (List.map Subtype.val l).prod
  -/
  rw [← coe_list_prod]
  /-
    case intro
    M : Type u_1
    B : Type u_3
    inst✝² : Monoid M
    inst✝¹ : SetLike B M
    inst✝ : SubmonoidClass B M
    S : B
    l : List (Subtype fun x => Membership.mem S x)
    ⊢ Membership.mem S ↑l.prod
  -/
  exact l.prod.coe_prop
  /-
    🎉 no goals
  -/


/-- Product of a multiset of elements in a submonoid of a `CommMonoid` is in the submonoid. -/
@[to_additive
      "Sum of a multiset of elements in an `AddSubmonoid` of an `AddCommMonoid` is
      in the `AddSubmonoid`."]
theorem multiset_prod_mem {M} [CommMonoid M] [SetLike B M] [SubmonoidClass B M] (m : Multiset M)
    (hm : ∀ a ∈ m, a ∈ S) : m.prod ∈ S := by
  /-
    B : Type u_3
    S : B
    M : Type u_4
    inst✝² : CommMonoid M
    inst✝¹ : SetLike B M
    inst✝ : SubmonoidClass B M
    m : Multiset M
    hm : ∀ (a : M), Membership.mem m a → Membership.mem S a
    ⊢ Membership.mem S m.prod
  -/
  lift m to Multiset S using hm
  /-
    case intro
    B : Type u_3
    S : B
    M : Type u_4
    inst✝² : CommMonoid M
    inst✝¹ : SetLike B M
    inst✝ : SubmonoidClass B M
    m : Multiset (Subtype fun x => Membership.mem S x)
    ⊢ Membership.mem S (Multiset.map Subtype.val m).prod
  -/
  rw [← coe_multiset_prod]
  /-
    case intro
    B : Type u_3
    S : B
    M : Type u_4
    inst✝² : CommMonoid M
    inst✝¹ : SetLike B M
    inst✝ : SubmonoidClass B M
    m : Multiset (Subtype fun x => Membership.mem S x)
    ⊢ Membership.mem S ↑m.prod
  -/
  exact m.prod.coe_prop
  /-
    🎉 no goals
  -/


/-- Product of elements of a submonoid of a `CommMonoid` indexed by a `Finset` is in the
    submonoid. -/
@[to_additive
      "Sum of elements in an `AddSubmonoid` of an `AddCommMonoid` indexed by a `Finset`
      is in the `AddSubmonoid`."]
theorem prod_mem {M : Type*} [CommMonoid M] [SetLike B M] [SubmonoidClass B M] {ι : Type*}
    {t : Finset ι} {f : ι → M} (h : ∀ c ∈ t, f c ∈ S) : (∏ c ∈ t, f c) ∈ S :=
  multiset_prod_mem (t.1.map f) fun _x hx =>
    let ⟨i, hi, hix⟩ := Multiset.mem_map.1 hx
    hix ▸ h i hi


@[to_additive (attr := norm_cast)]
theorem coe_list_prod (l : List s) : (l.prod : M) = (l.map (↑)).prod :=
  map_list_prod s.subtype l


@[to_additive (attr := norm_cast)]
theorem coe_multiset_prod {M} [CommMonoid M] (S : Submonoid M) (m : Multiset S) :
    (m.prod : M) = (m.map (↑)).prod :=
  S.subtype.map_multiset_prod m


@[to_additive (attr := norm_cast)]
theorem coe_finset_prod {ι M} [CommMonoid M] (S : Submonoid M) (f : ι → S) (s : Finset ι) :
    ↑(∏ i ∈ s, f i) = (∏ i ∈ s, f i : M) :=
  map_prod S.subtype f s


/-- Product of a list of elements in a submonoid is in the submonoid. -/
@[to_additive "Sum of a list of elements in an `AddSubmonoid` is in the `AddSubmonoid`."]
theorem list_prod_mem {l : List M} (hl : ∀ x ∈ l, x ∈ s) : l.prod ∈ s := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    s : Submonoid M
    l : List M
    hl : ∀ (x : M), Membership.mem l x → Membership.mem s x
    ⊢ Membership.mem s l.prod
  -/
  lift l to List s using hl
  /-
    case intro
    M : Type u_1
    inst✝ : Monoid M
    s : Submonoid M
    l : List (Subtype fun x => Membership.mem s x)
    ⊢ Membership.mem s (List.map Subtype.val l).prod
  -/
  rw [← coe_list_prod]
  /-
    case intro
    M : Type u_1
    inst✝ : Monoid M
    s : Submonoid M
    l : List (Subtype fun x => Membership.mem s x)
    ⊢ Membership.mem s ↑l.prod
  -/
  exact l.prod.coe_prop
  /-
    🎉 no goals
  -/


/-- Product of a multiset of elements in a submonoid of a `CommMonoid` is in the submonoid. -/
@[to_additive
      "Sum of a multiset of elements in an `AddSubmonoid` of an `AddCommMonoid` is
      in the `AddSubmonoid`."]
theorem multiset_prod_mem {M} [CommMonoid M] (S : Submonoid M) (m : Multiset M)
    (hm : ∀ a ∈ m, a ∈ S) : m.prod ∈ S := by
  /-
    M : Type u_4
    inst✝ : CommMonoid M
    S : Submonoid M
    m : Multiset M
    hm : ∀ (a : M), Membership.mem m a → Membership.mem S a
    ⊢ Membership.mem S m.prod
  -/
  lift m to Multiset S using hm
  /-
    case intro
    M : Type u_4
    inst✝ : CommMonoid M
    S : Submonoid M
    m : Multiset (Subtype fun x => Membership.mem S x)
    ⊢ Membership.mem S (Multiset.map Subtype.val m).prod
  -/
  rw [← coe_multiset_prod]
  /-
    case intro
    M : Type u_4
    inst✝ : CommMonoid M
    S : Submonoid M
    m : Multiset (Subtype fun x => Membership.mem S x)
    ⊢ Membership.mem S ↑m.prod
  -/
  exact m.prod.coe_prop
  /-
    🎉 no goals
  -/


@[to_additive]
theorem multiset_noncommProd_mem (S : Submonoid M) (m : Multiset M) (comm) (h : ∀ x ∈ m, x ∈ S) :
    m.noncommProd comm ∈ S := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    S : Submonoid M
    m : Multiset M
    comm : (setOf fun x => Membership.mem m x).Pairwise Commute
    h : ∀ (x : M), Membership.mem m x → Membership.mem S x
    ⊢ Membership.mem S (m.noncommProd comm)
  -/
  induction m using Quotient.inductionOn with | h l => ?_
  /-
    case h
    M : Type u_1
    inst✝ : Monoid M
    S : Submonoid M
    l : List M
    comm : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid M) l) x).Pai …
    h : ∀ (x : M), Membership.mem (Quotient.mk (List.isSetoid M) l) x → Membership …
    ⊢ Membership.mem S (Multiset.noncommProd (Quotient.mk (List.isSetoid M) l) comm)
  -/
  simp only [Multiset.quot_mk_to_coe, Multiset.noncommProd_coe]
  /-
    case h
    M : Type u_1
    inst✝ : Monoid M
    S : Submonoid M
    l : List M
    comm : (setOf fun x => Membership.mem (Quotient.mk (List.isSetoid M) l) x).Pai …
    h : ∀ (x : M), Membership.mem (Quotient.mk (List.isSetoid M) l) x → Membership …
    ⊢ Membership.mem S l.prod
  -/
  exact Submonoid.list_prod_mem _ h
  /-
    🎉 no goals
  -/


/-- Product of elements of a submonoid of a `CommMonoid` indexed by a `Finset` is in the
    submonoid. -/
@[to_additive
      "Sum of elements in an `AddSubmonoid` of an `AddCommMonoid` indexed by a `Finset`
      is in the `AddSubmonoid`."]
theorem prod_mem {M : Type*} [CommMonoid M] (S : Submonoid M) {ι : Type*} {t : Finset ι}
    {f : ι → M} (h : ∀ c ∈ t, f c ∈ S) : (∏ c ∈ t, f c) ∈ S :=
  S.multiset_prod_mem (t.1.map f) fun _ hx =>
    let ⟨i, hi, hix⟩ := Multiset.mem_map.1 hx
    hix ▸ h i hi


@[to_additive]
theorem noncommProd_mem (S : Submonoid M) {ι : Type*} (t : Finset ι) (f : ι → M) (comm)
    (h : ∀ c ∈ t, f c ∈ S) : t.noncommProd f comm ∈ S := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    S : Submonoid M
    ι : Type u_4
    t : Finset ι
    f : ι → M
    comm : (↑t).Pairwise (Function.onFun Commute f)
    h : ∀ (c : ι), Membership.mem t c → Membership.mem S (f c)
    ⊢ Membership.mem S (t.noncommProd f comm)
  -/
  apply multiset_noncommProd_mem
  /-
    case h
    M : Type u_1
    inst✝ : Monoid M
    S : Submonoid M
    ι : Type u_4
    t : Finset ι
    f : ι → M
    comm : (↑t).Pairwise (Function.onFun Commute f)
    h : ∀ (c : ι), Membership.mem t c → Membership.mem S (f c)
    ⊢ ∀ (x : M), Membership.mem (Multiset.map f t.val) x → Membership.mem S x
  -/
  intro y
  /-
    case h
    M : Type u_1
    inst✝ : Monoid M
    S : Submonoid M
    ι : Type u_4
    t : Finset ι
    f : ι → M
    comm : (↑t).Pairwise (Function.onFun Commute f)
    h : ∀ (c : ι), Membership.mem t c → Membership.mem S (f c)
    y : M
    ⊢ Membership.mem (Multiset.map f t.val) y → Membership.mem S y
  -/
  rw [Multiset.mem_map]
  /-
    case h
    M : Type u_1
    inst✝ : Monoid M
    S : Submonoid M
    ι : Type u_4
    t : Finset ι
    f : ι → M
    comm : (↑t).Pairwise (Function.onFun Commute f)
    h : ∀ (c : ι), Membership.mem t c → Membership.mem S (f c)
    y : M
    ⊢ (Exists fun a => And (Membership.mem t.val a) (Eq (f a) y)) → Membership.mem …
  -/
  rintro ⟨x, ⟨hx, rfl⟩⟩
  /-
    case h.intro.intro
    M : Type u_1
    inst✝ : Monoid M
    S : Submonoid M
    ι : Type u_4
    t : Finset ι
    f : ι → M
    comm : (↑t).Pairwise (Function.onFun Commute f)
    h : ∀ (c : ι), Membership.mem t c → Membership.mem S (f c)
    x : ι
    hx : Membership.mem t.val x
    ⊢ Membership.mem S (f x)
  -/
  exact h x hx
  /-
    🎉 no goals
  -/


