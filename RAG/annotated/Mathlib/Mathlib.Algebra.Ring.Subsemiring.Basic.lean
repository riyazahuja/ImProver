instance instCharZero [CharZero R] : CharZero s :=
  ⟨Function.Injective.of_comp (f := Subtype.val) (g := Nat.cast (R := s)) Nat.cast_injective⟩


@[mono]
theorem toSubmonoid_strictMono : StrictMono (toSubmonoid : Subsemiring R → Submonoid R) :=
  fun _ _ => id


@[mono]
theorem toSubmonoid_mono : Monotone (toSubmonoid : Subsemiring R → Submonoid R) :=
  toSubmonoid_strictMono.monotone


@[mono]
theorem toAddSubmonoid_strictMono : StrictMono (toAddSubmonoid : Subsemiring R → AddSubmonoid R) :=
  fun _ _ => id


@[mono]
theorem toAddSubmonoid_mono : Monotone (toAddSubmonoid : Subsemiring R → AddSubmonoid R) :=
  toAddSubmonoid_strictMono.monotone


/-- Product of a list of elements in a `Subsemiring` is in the `Subsemiring`. -/
nonrec theorem list_prod_mem {R : Type*} [Semiring R] (s : Subsemiring R) {l : List R} :
    (∀ x ∈ l, x ∈ s) → l.prod ∈ s :=
  list_prod_mem


/-- Sum of a list of elements in a `Subsemiring` is in the `Subsemiring`. -/
protected theorem list_sum_mem {l : List R} : (∀ x ∈ l, x ∈ s) → l.sum ∈ s :=
  list_sum_mem


/-- Product of a multiset of elements in a `Subsemiring` of a `CommSemiring`
    is in the `Subsemiring`. -/
protected theorem multiset_prod_mem {R} [CommSemiring R] (s : Subsemiring R) (m : Multiset R) :
    (∀ a ∈ m, a ∈ s) → m.prod ∈ s :=
  multiset_prod_mem m


/-- Sum of a multiset of elements in a `Subsemiring` of a `Semiring` is
in the `add_subsemiring`. -/
protected theorem multiset_sum_mem (m : Multiset R) : (∀ a ∈ m, a ∈ s) → m.sum ∈ s :=
  multiset_sum_mem m


/-- Product of elements of a subsemiring of a `CommSemiring` indexed by a `Finset` is in the
    subsemiring. -/
protected theorem prod_mem {R : Type*} [CommSemiring R] (s : Subsemiring R) {ι : Type*}
    {t : Finset ι} {f : ι → R} (h : ∀ c ∈ t, f c ∈ s) : (∏ i ∈ t, f i) ∈ s :=
  prod_mem h


/-- Sum of elements in a `Subsemiring` of a `Semiring` indexed by a `Finset`
is in the `add_subsemiring`. -/
protected theorem sum_mem (s : Subsemiring R) {ι : Type*} {t : Finset ι} {f : ι → R}
    (h : ∀ c ∈ t, f c ∈ s) : (∑ i ∈ t, f i) ∈ s :=
  sum_mem h


/-- The ring equiv between the top element of `Subsemiring R` and `R`. -/
@[simps]
def topEquiv : (⊤ : Subsemiring R) ≃+* R where
  toFun r := r
  invFun r := ⟨r, Subsemiring.mem_top r⟩
  left_inv _ := rfl
  right_inv _ := rfl
  map_mul' := (⊤ : Subsemiring R).coe_mul
  map_add' := (⊤ : Subsemiring R).coe_add


/-- The preimage of a subsemiring along a ring homomorphism is a subsemiring. -/
@[simps coe toSubmonoid]
def comap (f : R →+* S) (s : Subsemiring S) : Subsemiring R :=
  { s.toSubmonoid.comap (f : R →* S), s.toAddSubmonoid.comap (f : R →+ S) with carrier := f ⁻¹' s }


@[simp]
theorem mem_comap {s : Subsemiring S} {f : R →+* S} {x : R} : x ∈ s.comap f ↔ f x ∈ s :=
  Iff.rfl


theorem comap_comap (s : Subsemiring T) (g : S →+* T) (f : R →+* S) :
    (s.comap g).comap f = s.comap (g.comp f) :=
  rfl


/-- The image of a subsemiring along a ring homomorphism is a subsemiring. -/
@[simps coe toSubmonoid]
def map (f : R →+* S) (s : Subsemiring R) : Subsemiring S :=
  { s.toSubmonoid.map (f : R →* S), s.toAddSubmonoid.map (f : R →+ S) with carrier := f '' s }


@[simp]
lemma mem_map {f : R →+* S} {s : Subsemiring R} {y : S} : y ∈ s.map f ↔ ∃ x ∈ s, f x = y := Iff.rfl


@[simp]
theorem map_id : s.map (RingHom.id R) = s :=
  SetLike.coe_injective <| Set.image_id _


theorem map_map (g : S →+* T) (f : R →+* S) : (s.map f).map g = s.map (g.comp f) :=
  SetLike.coe_injective <| Set.image_image _ _ _


theorem map_le_iff_le_comap {f : R →+* S} {s : Subsemiring R} {t : Subsemiring S} :
    s.map f ≤ t ↔ s ≤ t.comap f :=
  Set.image_subset_iff


theorem gc_map_comap (f : R →+* S) : GaloisConnection (map f) (comap f) := fun _ _ =>
  map_le_iff_le_comap


/-- A subsemiring is isomorphic to its image under an injective function -/
noncomputable def equivMapOfInjective (f : R →+* S) (hf : Function.Injective f) : s ≃+* s.map f :=
  { Equiv.Set.image f s hf with
    map_mul' := fun _ _ => Subtype.ext (f.map_mul _ _)
    map_add' := fun _ _ => Subtype.ext (f.map_add _ _) }


@[simp]
theorem coe_equivMapOfInjective_apply (f : R →+* S) (hf : Function.Injective f) (x : s) :
    (equivMapOfInjective s f hf x : S) = f x :=
  rfl


/-- The range of a ring homomorphism is a subsemiring. See Note [range copy pattern]. -/
@[simps! coe toSubmonoid]
def rangeS : Subsemiring S :=
  ((⊤ : Subsemiring R).map f).copy (Set.range f) Set.image_univ.symm


@[simp]
theorem mem_rangeS {f : R →+* S} {y : S} : y ∈ f.rangeS ↔ ∃ x, f x = y :=
  Iff.rfl


theorem rangeS_eq_map (f : R →+* S) : f.rangeS = (⊤ : Subsemiring R).map f := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : NonAssocSemiring R
    inst✝ : NonAssocSemiring S
    f : RingHom R S
    ⊢ Eq f.rangeS (Subsemiring.map f Top.top)
  -/
  ext
  /-
    case h
    R : Type u
    S : Type v
    inst✝¹ : NonAssocSemiring R
    inst✝ : NonAssocSemiring S
    f : RingHom R S
    x✝ : S
    ⊢ Iff (Membership.mem f.rangeS x✝) (Membership.mem (Subsemiring.map f Top.top) …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mem_rangeS_self (f : R →+* S) (x : R) : f x ∈ f.rangeS :=
  mem_rangeS.mpr ⟨x, rfl⟩


theorem map_rangeS : f.rangeS.map g = (g.comp f).rangeS := by
  /-
    R : Type u
    S : Type v
    T : Type w
    inst✝² : NonAssocSemiring R
    inst✝¹ : NonAssocSemiring S
    inst✝ : NonAssocSemiring T
    g : RingHom S T
    f : RingHom R S
    ⊢ Eq (Subsemiring.map g f.rangeS) (g.comp f).rangeS
  -/
  simpa only [rangeS_eq_map] using (⊤ : Subsemiring R).map_map g f
  /-
    🎉 no goals
  -/


/-- The range of a morphism of semirings is a fintype, if the domain is a fintype.
Note: this instance can form a diamond with `Subtype.fintype` in the
  presence of `Fintype S`. -/
instance fintypeRangeS [Fintype R] [DecidableEq S] (f : R →+* S) : Fintype (rangeS f) :=
  Set.fintypeRange f


instance : Bot (Subsemiring R) :=
  ⟨(Nat.castRingHom R).rangeS⟩


instance : Inhabited (Subsemiring R) :=
  ⟨⊥⟩


theorem coe_bot : ((⊥ : Subsemiring R) : Set R) = Set.range ((↑) : ℕ → R) :=
  (Nat.castRingHom R).coe_rangeS


theorem mem_bot {x : R} : x ∈ (⊥ : Subsemiring R) ↔ ∃ n : ℕ, ↑n = x :=
  RingHom.mem_rangeS


instance : InfSet (Subsemiring R) :=
  ⟨fun s =>
                                                                           /-
                                                                             R : Type u
                                                                             S : Type v
                                                                             T : Type w
                                                                             inst✝² : NonAssocSemiring R
                                                                             M : Submonoid R
                                                                             inst✝¹ : NonAssocSemiring S
                                                                             inst✝ : NonAssocSemiring T
                                                                             s : Set (Subsemiring R)
                                                                             ⊢ Eq (↑(iInf fun t => iInf fun h => t.toSubmonoid)) (Set.iInter fun t => Set.i …
                                                                           -/
    Subsemiring.mk' (⋂ t ∈ s, ↑t) (⨅ t ∈ s, Subsemiring.toSubmonoid t) (by simp)
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
      (⨅ t ∈ s, Subsemiring.toAddSubmonoid t)
          /-
            R : Type u
            S : Type v
            T : Type w
            inst✝² : NonAssocSemiring R
            M : Submonoid R
            inst✝¹ : NonAssocSemiring S
            inst✝ : NonAssocSemiring T
            s : Set (Subsemiring R)
            ⊢ Eq (↑(iInf fun t => iInf fun h => t.toAddSubmonoid)) (Set.iInter fun t => Se …
          -/
      (by simp)⟩
          /-
            🎉 no goals
          -/


@[simp, norm_cast]
theorem coe_sInf (S : Set (Subsemiring R)) : ((sInf S : Subsemiring R) : Set R) = ⋂ s ∈ S, ↑s :=
  rfl


theorem mem_sInf {S : Set (Subsemiring R)} {x : R} : x ∈ sInf S ↔ ∀ p ∈ S, x ∈ p :=
  Set.mem_iInter₂


@[simp, norm_cast]
theorem coe_iInf {ι : Sort*} {S : ι → Subsemiring R} : (↑(⨅ i, S i) : Set R) = ⋂ i, S i := by
  /-
    R : Type u
    inst✝ : NonAssocSemiring R
    ι : Sort u_1
    S : ι → Subsemiring R
    ⊢ Eq (↑(iInf fun i => S i)) (Set.iInter fun i => ↑(S i))
  -/
  simp only [iInf, coe_sInf, Set.biInter_range]
  /-
    🎉 no goals
  -/


theorem mem_iInf {ι : Sort*} {S : ι → Subsemiring R} {x : R} : (x ∈ ⨅ i, S i) ↔ ∀ i, x ∈ S i := by
  /-
    R : Type u
    inst✝ : NonAssocSemiring R
    ι : Sort u_1
    S : ι → Subsemiring R
    x : R
    ⊢ Iff (Membership.mem (iInf fun i => S i) x) (∀ (i : ι), Membership.mem (S i) x)
  -/
  simp only [iInf, mem_sInf, Set.forall_mem_range]
  /-
    🎉 no goals
  -/


@[simp]
theorem sInf_toSubmonoid (s : Set (Subsemiring R)) :
    (sInf s).toSubmonoid = ⨅ t ∈ s, Subsemiring.toSubmonoid t :=
  mk'_toSubmonoid _ _


@[simp]
theorem sInf_toAddSubmonoid (s : Set (Subsemiring R)) :
    (sInf s).toAddSubmonoid = ⨅ t ∈ s, Subsemiring.toAddSubmonoid t :=
  mk'_toAddSubmonoid _ _


/-- Subsemirings of a semiring form a complete lattice. -/
instance : CompleteLattice (Subsemiring R) :=
  { completeLatticeOfInf (Subsemiring R) fun _ =>
      IsGLB.of_image
        (fun {s t : Subsemiring R} => show (s : Set R) ⊆ t ↔ s ≤ t from SetLike.coe_subset_coe)
        isGLB_biInf with
    bot := ⊥
    bot_le := fun s _ hx =>
      let ⟨n, hn⟩ := mem_bot.1 hx
      hn ▸ natCast_mem s n
    top := ⊤
    le_top := fun _ _ _ => trivial
    inf := (· ⊓ ·)
    inf_le_left := fun _ _ _ => And.left
    inf_le_right := fun _ _ _ => And.right
    le_inf := fun _ _ _ h₁ h₂ _ hx => ⟨h₁ hx, h₂ hx⟩ }


theorem eq_top_iff' (A : Subsemiring R) : A = ⊤ ↔ ∀ x : R, x ∈ A :=
  eq_top_iff.trans ⟨fun h m => h <| mem_top m, fun h m _ => h m⟩


/-- The center of a non-associative semiring `R` is the set of elements that commute and associate
with everything in `R` -/
@[simps coe toSubmonoid]
def center : Subsemiring R :=
  { NonUnitalSubsemiring.center R with
    one_mem' := Set.one_mem_center }


/-- The center is commutative and associative.

This is not an instance as it forms a non-defeq diamond with
`NonUnitalSubringClass.tNonUnitalring ` in the `npow` field. -/
abbrev center.commSemiring' : CommSemiring (center R) :=
  { Submonoid.center.commMonoid', (center R).toNonAssocSemiring with }


/-- The center is commutative. -/
instance center.commSemiring {R} [Semiring R] : CommSemiring (center R) :=
  { Submonoid.center.commMonoid, (center R).toSemiring with }

-- no instance diamond, unlike the primed version

theorem mem_center_iff {R} [Semiring R] {z : R} : z ∈ center R ↔ ∀ g, g * z = z * g :=
  Subsemigroup.mem_center_iff


instance decidableMemCenter {R} [Semiring R] [DecidableEq R] [Fintype R] :
    DecidablePred (· ∈ center R) := fun _ => decidable_of_iff' _ mem_center_iff


@[simp]
theorem center_eq_top (R) [CommSemiring R] : center R = ⊤ :=
  SetLike.coe_injective (Set.center_eq_univ R)



/-- The centralizer of a set as subsemiring. -/
def centralizer {R} [Semiring R] (s : Set R) : Subsemiring R :=
  { Submonoid.centralizer s with
    carrier := s.centralizer
    zero_mem' := Set.zero_mem_centralizer
    add_mem' := Set.add_mem_centralizer }


@[simp, norm_cast]
theorem coe_centralizer {R} [Semiring R] (s : Set R) : (centralizer s : Set R) = s.centralizer :=
  rfl


theorem centralizer_toSubmonoid {R} [Semiring R] (s : Set R) :
    (centralizer s).toSubmonoid = Submonoid.centralizer s :=
  rfl


theorem mem_centralizer_iff {R} [Semiring R] {s : Set R} {z : R} :
    z ∈ centralizer s ↔ ∀ g ∈ s, g * z = z * g :=
  Iff.rfl


theorem center_le_centralizer {R} [Semiring R] (s) : center R ≤ centralizer s :=
  s.center_subset_centralizer


theorem centralizer_le {R} [Semiring R] (s t : Set R) (h : s ⊆ t) : centralizer t ≤ centralizer s :=
  Set.centralizer_subset h


@[simp]
theorem centralizer_eq_top_iff_subset {R} [Semiring R] {s : Set R} :
    centralizer s = ⊤ ↔ s ⊆ center R :=
  SetLike.ext'_iff.trans Set.centralizer_eq_top_iff_subset


@[simp]
theorem centralizer_univ {R} [Semiring R] : centralizer Set.univ = center R :=
  SetLike.ext' (Set.centralizer_univ R)


lemma le_centralizer_centralizer {R} [Semiring R] {s : Subsemiring R} :
    s ≤ centralizer (centralizer (s : Set R)) :=
  Set.subset_centralizer_centralizer


@[simp]
lemma centralizer_centralizer_centralizer {R} [Semiring R] {s : Set R} :
    centralizer s.centralizer.centralizer = centralizer s := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    s : Set R
    ⊢ Eq (Subsemiring.centralizer s.centralizer.centralizer) (Subsemiring.centrali …
  -/
  apply SetLike.coe_injective
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    s : Set R
    ⊢ Eq ↑(Subsemiring.centralizer s.centralizer.centralizer) ↑(Subsemiring.centra …
  -/
  simp only [coe_centralizer, Set.centralizer_centralizer_centralizer]
  /-
    🎉 no goals
  -/


/-- The `Subsemiring` generated by a set. -/
def closure (s : Set R) : Subsemiring R :=
  sInf { S | s ⊆ S }


theorem mem_closure {x : R} {s : Set R} : x ∈ closure s ↔ ∀ S : Subsemiring R, s ⊆ S → x ∈ S :=
  mem_sInf


/-- The subsemiring generated by a set includes the set. -/
@[simp, aesop safe 20 apply (rule_sets := [SetLike])]
theorem subset_closure {s : Set R} : s ⊆ closure s := fun _ hx => mem_closure.2 fun _ hS => hS hx


theorem not_mem_of_not_mem_closure {s : Set R} {P : R} (hP : P ∉ closure s) : P ∉ s := fun h =>
  hP (subset_closure h)


/-- A subsemiring `S` includes `closure s` if and only if it includes `s`. -/
@[simp]
theorem closure_le {s : Set R} {t : Subsemiring R} : closure s ≤ t ↔ s ⊆ t :=
  ⟨Set.Subset.trans subset_closure, fun h => sInf_le h⟩


/-- Subsemiring closure of a set is monotone in its argument: if `s ⊆ t`,
then `closure s ≤ closure t`. -/
@[gcongr]
theorem closure_mono ⦃s t : Set R⦄ (h : s ⊆ t) : closure s ≤ closure t :=
  closure_le.2 <| Set.Subset.trans h subset_closure


theorem closure_eq_of_le {s : Set R} {t : Subsemiring R} (h₁ : s ⊆ t) (h₂ : t ≤ closure s) :
    closure s = t :=
  le_antisymm (closure_le.2 h₁) h₂


theorem mem_map_equiv {f : R ≃+* S} {K : Subsemiring R} {x : S} :
    x ∈ K.map (f : R →+* S) ↔ f.symm x ∈ K := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : NonAssocSemiring R
    inst✝ : NonAssocSemiring S
    f : RingEquiv R S
    K : Subsemiring R
    x : S
    ⊢ Iff (Membership.mem (Subsemiring.map (↑f) K) x) (Membership.mem K (f.symm x))
  -/
  convert @Set.mem_image_equiv _ _ (↑K) f.toEquiv x using 1
  /-
    🎉 no goals
  -/


theorem map_equiv_eq_comap_symm (f : R ≃+* S) (K : Subsemiring R) :
    K.map (f : R →+* S) = K.comap f.symm :=
  SetLike.coe_injective (f.toEquiv.image_eq_preimage K)


theorem comap_equiv_eq_map_symm (f : R ≃+* S) (K : Subsemiring S) :
    K.comap (f : R →+* S) = K.map f.symm :=
  (map_equiv_eq_comap_symm f.symm K).symm


/-- The additive closure of a submonoid is a subsemiring. -/
def subsemiringClosure (M : Submonoid R) : Subsemiring R :=
  { AddSubmonoid.closure (M : Set R) with
    one_mem' := AddSubmonoid.mem_closure.mpr fun _ hy => hy M.one_mem
    mul_mem' := MulMemClass.mul_mem_add_closure }


theorem subsemiringClosure_coe :
    (M.subsemiringClosure : Set R) = AddSubmonoid.closure (M : Set R) :=
  rfl


theorem subsemiringClosure_mem {x : R} :
    x ∈ M.subsemiringClosure ↔ x ∈ AddSubmonoid.closure (M : Set R) :=
  Iff.rfl


theorem subsemiringClosure_toAddSubmonoid :
    M.subsemiringClosure.toAddSubmonoid = AddSubmonoid.closure (M : Set R) :=
  rfl


/-- The `Subsemiring` generated by a multiplicative submonoid coincides with the
`Subsemiring.closure` of the submonoid itself . -/
theorem subsemiringClosure_eq_closure : M.subsemiringClosure = Subsemiring.closure (M : Set R) := by
  /-
    R : Type u
    inst✝ : NonAssocSemiring R
    M : Submonoid R
    ⊢ Eq M.subsemiringClosure (Subsemiring.closure ↑M)
  -/
  ext
  refine
    ⟨fun hx => ?_, fun hx =>
      (Subsemiring.mem_closure.mp hx) M.subsemiringClosure fun s sM => ?_⟩
      /-
        case h.refine_1
        R : Type u
        inst✝ : NonAssocSemiring R
        M : Submonoid R
        x✝ : R
        hx : Membership.mem M.subsemiringClosure x✝
        ⊢ Membership.mem (Subsemiring.closure ↑M) x✝
      -/
  <;> rintro - ⟨H1, rfl⟩
      /-
        case h.refine_1.intro
        R : Type u
        inst✝ : NonAssocSemiring R
        M : Submonoid R
        x✝ : R
        hx : Membership.mem M.subsemiringClosure x✝
        H1 : Subsemiring R
        ⊢ Membership.mem ((fun t => Set.iInter fun h => ↑t) H1) x✝
      -/
  <;> rintro - ⟨H2, rfl⟩
    /-
      case h.refine_1.intro.intro
      R : Type u
      inst✝ : NonAssocSemiring R
      M : Submonoid R
      x✝ : R
      hx : Membership.mem M.subsemiringClosure x✝
      H1 : Subsemiring R
      H2 : Membership.mem (setOf fun S => HasSubset.Subset ↑M ↑S) H1
      ⊢ Membership.mem ((fun h => ↑H1) H2) x✝
    -/
  · exact AddSubmonoid.mem_closure.mp hx H1.toAddSubmonoid H2
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2.intro.intro
      R : Type u
      inst✝ : NonAssocSemiring R
      M : Submonoid R
      x✝ : R
      hx : Membership.mem (Subsemiring.closure ↑M) x✝
      s : R
      sM : Membership.mem (↑M) s
      H1 : AddSubmonoid R
      H2 : Membership.mem (setOf fun S => HasSubset.Subset ↑M ↑S) H1
      ⊢ Membership.mem ((fun h => ↑H1) H2) s
    -/
  · exact H2 sM
    /-
      🎉 no goals
    -/


@[simp]
theorem closure_submonoid_closure (s : Set R) : closure ↑(Submonoid.closure s) = closure s :=
  le_antisymm
    (closure_le.mpr fun _ hy =>
      (Submonoid.mem_closure.mp hy) (closure s).toSubmonoid subset_closure)
    (closure_mono Submonoid.subset_closure)


/-- The elements of the subsemiring closure of `M` are exactly the elements of the additive closure
of a multiplicative submonoid `M`. -/
theorem coe_closure_eq (s : Set R) :
    (closure s : Set R) = AddSubmonoid.closure (Submonoid.closure s : Set R) := by
  /-
    R : Type u
    inst✝ : NonAssocSemiring R
    s : Set R
    ⊢ Eq ↑(Subsemiring.closure s) ↑(AddSubmonoid.closure ↑(Submonoid.closure s))
  -/
  simp [← Submonoid.subsemiringClosure_toAddSubmonoid, Submonoid.subsemiringClosure_eq_closure]
  /-
    🎉 no goals
  -/


theorem mem_closure_iff {s : Set R} {x} :
    x ∈ closure s ↔ x ∈ AddSubmonoid.closure (Submonoid.closure s : Set R) :=
  Set.ext_iff.mp (coe_closure_eq s) x


@[simp]
theorem closure_addSubmonoid_closure {s : Set R} :
    closure ↑(AddSubmonoid.closure s) = closure s := by
  /-
    R : Type u
    inst✝ : NonAssocSemiring R
    s : Set R
    ⊢ Eq (Subsemiring.closure ↑(AddSubmonoid.closure s)) (Subsemiring.closure s)
  -/
  ext x
  /-
    case h
    R : Type u
    inst✝ : NonAssocSemiring R
    s : Set R
    x : R
    ⊢ Iff (Membership.mem (Subsemiring.closure ↑(AddSubmonoid.closure s)) x) (Memb …
  -/
  refine ⟨fun hx => ?_, fun hx => closure_mono AddSubmonoid.subset_closure hx⟩
  /-
    case h
    R : Type u
    inst✝ : NonAssocSemiring R
    s : Set R
    x : R
    hx : Membership.mem (Subsemiring.closure ↑(AddSubmonoid.closure s)) x
    ⊢ Membership.mem (Subsemiring.closure s) x
  -/
  rintro - ⟨H, rfl⟩
  /-
    case h.intro
    R : Type u
    inst✝ : NonAssocSemiring R
    s : Set R
    x : R
    hx : Membership.mem (Subsemiring.closure ↑(AddSubmonoid.closure s)) x
    H : Subsemiring R
    ⊢ Membership.mem ((fun t => Set.iInter fun h => ↑t) H) x
  -/
  rintro - ⟨J, rfl⟩
  /-
    case h.intro.intro
    R : Type u
    inst✝ : NonAssocSemiring R
    s : Set R
    x : R
    hx : Membership.mem (Subsemiring.closure ↑(AddSubmonoid.closure s)) x
    H : Subsemiring R
    J : Membership.mem (setOf fun S => HasSubset.Subset s ↑S) H
    ⊢ Membership.mem ((fun h => ↑H) J) x
  -/
  refine (AddSubmonoid.mem_closure.mp (mem_closure_iff.mp hx)) H.toAddSubmonoid fun y hy => ?_
  /-
    case h.intro.intro
    R : Type u
    inst✝ : NonAssocSemiring R
    s : Set R
    x : R
    hx : Membership.mem (Subsemiring.closure ↑(AddSubmonoid.closure s)) x
    H : Subsemiring R
    J : Membership.mem (setOf fun S => HasSubset.Subset s ↑S) H
    y : R
    hy : Membership.mem (↑(Submonoid.closure ↑(AddSubmonoid.closure s))) y
    ⊢ Membership.mem (↑H.toAddSubmonoid) y
  -/
  refine (Submonoid.mem_closure.mp hy) H.toSubmonoid fun z hz => ?_
  /-
    case h.intro.intro
    R : Type u
    inst✝ : NonAssocSemiring R
    s : Set R
    x : R
    hx : Membership.mem (Subsemiring.closure ↑(AddSubmonoid.closure s)) x
    H : Subsemiring R
    J : Membership.mem (setOf fun S => HasSubset.Subset s ↑S) H
    y : R
    hy : Membership.mem (↑(Submonoid.closure ↑(AddSubmonoid.closure s))) y
    z : R
    hz : Membership.mem (↑(AddSubmonoid.closure s)) z
    ⊢ Membership.mem (↑H.toSubmonoid) z
  -/
  exact (AddSubmonoid.mem_closure.mp hz) H.toAddSubmonoid fun w hw => J hw
  /-
    🎉 no goals
  -/


/-- An induction principle for closure membership. If `p` holds for `0`, `1`, and all elements
of `s`, and is preserved under addition and multiplication, then `p` holds for all elements
of the closure of `s`. -/
@[elab_as_elim]
theorem closure_induction {s : Set R} {p : (x : R) → x ∈ closure s → Prop}
    (mem : ∀ (x) (hx : x ∈ s), p x (subset_closure hx))
    (zero : p 0 (zero_mem _)) (one : p 1 (one_mem _))
    (add : ∀ x y hx hy, p x hx → p y hy → p (x + y) (add_mem hx hy))
    (mul : ∀ x y hx hy, p x hx → p y hy → p (x * y) (mul_mem hx hy))
    {x} (hx : x ∈ closure s)  : p x hx :=
  let K : Subsemiring R :=
    { carrier := { x | ∃ hx, p x hx }
      mul_mem' := fun ⟨_, hpx⟩ ⟨_, hpy⟩ ↦ ⟨_, mul _ _ _ _ hpx hpy⟩
      add_mem' := fun ⟨_, hpx⟩ ⟨_, hpy⟩ ↦ ⟨_, add _ _ _ _ hpx hpy⟩
      one_mem' := ⟨_, one⟩
      zero_mem' := ⟨_, zero⟩ }
  closure_le (t := K) |>.mpr (fun y hy ↦ ⟨subset_closure hy, mem y hy⟩) hx |>.elim fun _ ↦ id


@[deprecated closure_induction (since := "2024-10-10")]
alias closure_induction' := closure_induction


/-- An induction principle for closure membership for predicates with two arguments. -/
@[elab_as_elim]
theorem closure_induction₂ {s : Set R} {p : (x y : R) → x ∈ closure s → y ∈ closure s → Prop}
    (mem_mem : ∀ (x) (y) (hx : x ∈ s) (hy : y ∈ s), p x y (subset_closure hx) (subset_closure hy))
    (zero_left : ∀ x hx, p 0 x (zero_mem _) hx) (zero_right : ∀ x hx, p x 0 hx (zero_mem _))
    (one_left : ∀ x hx, p 1 x (one_mem _) hx) (one_right : ∀ x hx, p x 1 hx (one_mem _))
    (add_left : ∀ x y z hx hy hz, p x z hx hz → p y z hy hz → p (x + y) z (add_mem hx hy) hz)
    (add_right : ∀ x y z hx hy hz, p x y hx hy → p x z hx hz → p x (y + z) hx (add_mem hy hz))
    (mul_left : ∀ x y z hx hy hz, p x z hx hz → p y z hy hz → p (x * y) z (mul_mem hx hy) hz)
    (mul_right : ∀ x y z hx hy hz, p x y hx hy → p x z hx hz → p x (y * z) hx (mul_mem hy hz))
    {x y : R} (hx : x ∈ closure s) (hy : y ∈ closure s) :
    p x y hx hy := by
  induction hy using closure_induction with
  | mem z hz => induction hx using closure_induction with
    | mem _ h => exact mem_mem _ _ h hz
    | zero => exact zero_left _ _
    | one => exact one_left _ _
    | mul _ _ _ _ h₁ h₂ => exact mul_left _ _ _ _ _ _ h₁ h₂
    | add _ _ _ _ h₁ h₂ => exact add_left _ _ _ _ _ _ h₁ h₂
  | zero => exact zero_right x hx
  | one => exact one_right x hx
  | mul _ _ _ _ h₁ h₂ => exact mul_right _ _ _ _ _ _ h₁ h₂
  | add _ _ _ _ h₁ h₂ => exact add_right _ _ _ _ _ _ h₁ h₂


theorem mem_closure_iff_exists_list {R} [Semiring R] {s : Set R} {x} :
    x ∈ closure s ↔ ∃ L : List (List R), (∀ t ∈ L, ∀ y ∈ t, y ∈ s) ∧ (L.map List.prod).sum = x := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    s : Set R
    x : R
    ⊢ Iff (Membership.mem (Subsemiring.closure s) x) (Exists fun L => And (∀ (t :  …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝ : Semiring R
      s : Set R
      x : R
      ⊢ Membership.mem (Subsemiring.closure s) x → Exists fun L => And (∀ (t : List  …
    -/
  · intro hx
    /-
      case mp
      R : Type u_1
      inst✝ : Semiring R
      s : Set R
      x : R
      hx : Membership.mem (Subsemiring.closure s) x
      ⊢ Exists fun L => And (∀ (t : List R), Membership.mem L t → ∀ (y : R), Members …
    -/
    rw [mem_closure_iff] at hx
    induction hx using AddSubmonoid.closure_induction with
    | mem x hx =>
      suffices ∃ t : List R, (∀ y ∈ t, y ∈ s) ∧ t.prod = x from
        let ⟨t, ht1, ht2⟩ := this
        ⟨[t], List.forall_mem_singleton.2 ht1, by
          rw [List.map_singleton, List.sum_singleton, ht2]⟩
      induction hx using Submonoid.closure_induction with
      | mem x hx => exact ⟨[x], List.forall_mem_singleton.2 hx, List.prod_singleton⟩
      | one => exact ⟨[], List.forall_mem_nil _, rfl⟩
      | mul x y _ _ ht hu =>
        obtain ⟨⟨t, ht1, ht2⟩, ⟨u, hu1, hu2⟩⟩ := And.intro ht hu
        exact ⟨t ++ u, List.forall_mem_append.2 ⟨ht1, hu1⟩, by rw [List.prod_append, ht2, hu2]⟩
    | one => exact ⟨[], List.forall_mem_nil _, rfl⟩
    | mul x y _ _ hL hM =>
      obtain ⟨⟨L, HL1, HL2⟩, ⟨M, HM1, HM2⟩⟩ := And.intro hL hM
      exact ⟨L ++ M, List.forall_mem_append.2 ⟨HL1, HM1⟩, by
        rw [List.map_append, List.sum_append, HL2, HM2]⟩
    /-
      case mpr
      R : Type u_1
      inst✝ : Semiring R
      s : Set R
      x : R
      ⊢ (Exists fun L => And (∀ (t : List R), Membership.mem L t → ∀ (y : R), Member …
    -/
  · rintro ⟨L, HL1, HL2⟩
    exact HL2 ▸
      list_sum_mem fun r hr =>
        let ⟨t, ht1, ht2⟩ := List.mem_map.1 hr
        ht2 ▸ list_prod_mem _ fun y hy => subset_closure <| HL1 t ht1 y hy


/-- `closure` forms a Galois insertion with the coercion to set. -/
protected def gi : GaloisInsertion (@closure R _) (↑) where
  choice s _ := closure s
  gc _ _ := closure_le
  le_l_u _ := subset_closure
  choice_eq _ _ := rfl


/-- Closure of a subsemiring `S` equals `S`. -/
theorem closure_eq (s : Subsemiring R) : closure (s : Set R) = s :=
  (Subsemiring.gi R).l_u_eq s


@[simp]
theorem closure_empty : closure (∅ : Set R) = ⊥ :=
  (Subsemiring.gi R).gc.l_bot


@[simp]
theorem closure_univ : closure (Set.univ : Set R) = ⊤ :=
  @coe_top R _ ▸ closure_eq ⊤


theorem closure_union (s t : Set R) : closure (s ∪ t) = closure s ⊔ closure t :=
  (Subsemiring.gi R).gc.l_sup


theorem closure_iUnion {ι} (s : ι → Set R) : closure (⋃ i, s i) = ⨆ i, closure (s i) :=
  (Subsemiring.gi R).gc.l_iSup


theorem closure_sUnion (s : Set (Set R)) : closure (⋃₀ s) = ⨆ t ∈ s, closure t :=
  (Subsemiring.gi R).gc.l_sSup


theorem map_sup (s t : Subsemiring R) (f : R →+* S) : (s ⊔ t).map f = s.map f ⊔ t.map f :=
  (gc_map_comap f).l_sup


theorem map_iSup {ι : Sort*} (f : R →+* S) (s : ι → Subsemiring R) :
    (iSup s).map f = ⨆ i, (s i).map f :=
  (gc_map_comap f).l_iSup


theorem map_inf (s t : Subsemiring R) (f : R →+* S) (hf : Function.Injective f) :
    (s ⊓ t).map f = s.map f ⊓ t.map f := SetLike.coe_injective (Set.image_inter hf)


theorem map_iInf {ι : Sort*} [Nonempty ι] (f : R →+* S) (hf : Function.Injective f)
    (s : ι → Subsemiring R) : (iInf s).map f = ⨅ i, (s i).map f := by
  /-
    R : Type u
    S : Type v
    inst✝² : NonAssocSemiring R
    inst✝¹ : NonAssocSemiring S
    ι : Sort u_1
    inst✝ : Nonempty ι
    f : RingHom R S
    hf : Function.Injective ⇑f
    s : ι → Subsemiring R
    ⊢ Eq (Subsemiring.map f (iInf s)) (iInf fun i => Subsemiring.map f (s i))
  -/
  apply SetLike.coe_injective
  /-
    case a
    R : Type u
    S : Type v
    inst✝² : NonAssocSemiring R
    inst✝¹ : NonAssocSemiring S
    ι : Sort u_1
    inst✝ : Nonempty ι
    f : RingHom R S
    hf : Function.Injective ⇑f
    s : ι → Subsemiring R
    ⊢ Eq ↑(Subsemiring.map f (iInf s)) ↑(iInf fun i => Subsemiring.map f (s i))
  -/
  simpa using (Set.injOn_of_injective hf).image_iInter_eq (s := SetLike.coe ∘ s)
  /-
    🎉 no goals
  -/


theorem comap_inf (s t : Subsemiring S) (f : R →+* S) : (s ⊓ t).comap f = s.comap f ⊓ t.comap f :=
  (gc_map_comap f).u_inf


theorem comap_iInf {ι : Sort*} (f : R →+* S) (s : ι → Subsemiring S) :
    (iInf s).comap f = ⨅ i, (s i).comap f :=
  (gc_map_comap f).u_iInf


@[simp]
theorem map_bot (f : R →+* S) : (⊥ : Subsemiring R).map f = ⊥ :=
  (gc_map_comap f).l_bot


@[simp]
theorem comap_top (f : R →+* S) : (⊤ : Subsemiring S).comap f = ⊤ :=
  (gc_map_comap f).u_top


/-- Given `Subsemiring`s `s`, `t` of semirings `R`, `S` respectively, `s.prod t` is `s × t`
as a subsemiring of `R × S`. -/
def prod (s : Subsemiring R) (t : Subsemiring S) : Subsemiring (R × S) :=
  { s.toSubmonoid.prod t.toSubmonoid, s.toAddSubmonoid.prod t.toAddSubmonoid with
    carrier := s ×ˢ t }


@[norm_cast]
theorem coe_prod (s : Subsemiring R) (t : Subsemiring S) :
    (s.prod t : Set (R × S)) = (s : Set R) ×ˢ (t : Set S) :=
  rfl


theorem mem_prod {s : Subsemiring R} {t : Subsemiring S} {p : R × S} :
    p ∈ s.prod t ↔ p.1 ∈ s ∧ p.2 ∈ t :=
  Iff.rfl


@[gcongr, mono]
theorem prod_mono ⦃s₁ s₂ : Subsemiring R⦄ (hs : s₁ ≤ s₂) ⦃t₁ t₂ : Subsemiring S⦄ (ht : t₁ ≤ t₂) :
    s₁.prod t₁ ≤ s₂.prod t₂ :=
  Set.prod_mono hs ht


theorem prod_mono_right (s : Subsemiring R) : Monotone fun t : Subsemiring S => s.prod t :=
  prod_mono (le_refl s)


theorem prod_mono_left (t : Subsemiring S) : Monotone fun s : Subsemiring R => s.prod t :=
  fun _ _ hs => prod_mono hs (le_refl t)


theorem prod_top (s : Subsemiring R) : s.prod (⊤ : Subsemiring S) = s.comap (RingHom.fst R S) :=
                  /-
                    R : Type u
                    S : Type v
                    inst✝¹ : NonAssocSemiring R
                    inst✝ : NonAssocSemiring S
                    s : Subsemiring R
                    x : Prod R S
                    ⊢ Iff (Membership.mem (s.prod Top.top) x) (Membership.mem (Subsemiring.comap ( …
                  -/
  ext fun x => by simp [mem_prod, MonoidHom.coe_fst]
                  /-
                    🎉 no goals
                  -/


theorem top_prod (s : Subsemiring S) : (⊤ : Subsemiring R).prod s = s.comap (RingHom.snd R S) :=
                  /-
                    R : Type u
                    S : Type v
                    inst✝¹ : NonAssocSemiring R
                    inst✝ : NonAssocSemiring S
                    s : Subsemiring S
                    x : Prod R S
                    ⊢ Iff (Membership.mem (Top.top.prod s) x) (Membership.mem (Subsemiring.comap ( …
                  -/
  ext fun x => by simp [mem_prod, MonoidHom.coe_snd]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem top_prod_top : (⊤ : Subsemiring R).prod (⊤ : Subsemiring S) = ⊤ :=
  (top_prod _).trans <| comap_top _


/-- Product of subsemirings is isomorphic to their product as monoids. -/
def prodEquiv (s : Subsemiring R) (t : Subsemiring S) : s.prod t ≃+* s × t :=
  { Equiv.Set.prod (s : Set R) (t : Set S) with
    map_mul' := fun _ _ => rfl
    map_add' := fun _ _ => rfl }


theorem mem_iSup_of_directed {ι} [hι : Nonempty ι] {S : ι → Subsemiring R} (hS : Directed (· ≤ ·) S)
    {x : R} : (x ∈ ⨆ i, S i) ↔ ∃ i, x ∈ S i := by
  /-
    R : Type u
    inst✝ : NonAssocSemiring R
    ι : Sort u_1
    hι : Nonempty ι
    S : ι → Subsemiring R
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x : R
    ⊢ Iff (Membership.mem (iSup fun i => S i) x) (Exists fun i => Membership.mem ( …
  -/
  refine ⟨?_, fun ⟨i, hi⟩ ↦ le_iSup S i hi⟩
  let U : Subsemiring R :=
    Subsemiring.mk' (⋃ i, (S i : Set R))
      (⨆ i, (S i).toSubmonoid) (Submonoid.coe_iSup_of_directed hS)
      (⨆ i, (S i).toAddSubmonoid) (AddSubmonoid.coe_iSup_of_directed hS)
  -- Porting note: gave the hypothesis an explicit name because `@this` doesn't work
  /-
    R : Type u
    inst✝ : NonAssocSemiring R
    ι : Sort u_1
    hι : Nonempty ι
    S : ι → Subsemiring R
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x : R
    U : Subsemiring R := Subsemiring.mk' (Set.iUnion fun i => ↑(S i)) (iSup fun i  …
    ⊢ Membership.mem (iSup fun i => S i) x → Exists fun i => Membership.mem (S i) x
  -/
  suffices h : ⨆ i, S i ≤ U by simpa [U] using @h x
  /-
    R : Type u
    inst✝ : NonAssocSemiring R
    ι : Sort u_1
    hι : Nonempty ι
    S : ι → Subsemiring R
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x : R
    U : Subsemiring R := Subsemiring.mk' (Set.iUnion fun i => ↑(S i)) (iSup fun i  …
    ⊢ LE.le (iSup fun i => S i) U
  -/
  exact iSup_le fun i x hx ↦ Set.mem_iUnion.2 ⟨i, hx⟩
  /-
    🎉 no goals
  -/


theorem coe_iSup_of_directed {ι} [hι : Nonempty ι] {S : ι → Subsemiring R}
    (hS : Directed (· ≤ ·) S) : ((⨆ i, S i : Subsemiring R) : Set R) = ⋃ i, S i :=
                     /-
                       R : Type u
                       inst✝ : NonAssocSemiring R
                       ι : Sort u_1
                       hι : Nonempty ι
                       S : ι → Subsemiring R
                       hS : Directed (fun x1 x2 => LE.le x1 x2) S
                       x : R
                       ⊢ Iff (Membership.mem (↑(iSup fun i => S i)) x) (Membership.mem (Set.iUnion fu …
                     -/
  Set.ext fun x ↦ by simp [mem_iSup_of_directed hS]
                     /-
                       🎉 no goals
                     -/


theorem mem_sSup_of_directedOn {S : Set (Subsemiring R)} (Sne : S.Nonempty)
    (hS : DirectedOn (· ≤ ·) S) {x : R} : x ∈ sSup S ↔ ∃ s ∈ S, x ∈ s := by
  /-
    R : Type u
    inst✝ : NonAssocSemiring R
    S : Set (Subsemiring R)
    Sne : S.Nonempty
    hS : DirectedOn (fun x1 x2 => LE.le x1 x2) S
    x : R
    ⊢ Iff (Membership.mem (SupSet.sSup S) x) (Exists fun s => And (Membership.mem  …
  -/
  haveI : Nonempty S := Sne.to_subtype
  simp only [sSup_eq_iSup', mem_iSup_of_directed hS.directed_val, SetCoe.exists, Subtype.coe_mk,
    exists_prop]


theorem coe_sSup_of_directedOn {S : Set (Subsemiring R)} (Sne : S.Nonempty)
    (hS : DirectedOn (· ≤ ·) S) : (↑(sSup S) : Set R) = ⋃ s ∈ S, ↑s :=
                      /-
                        R : Type u
                        inst✝ : NonAssocSemiring R
                        S : Set (Subsemiring R)
                        Sne : S.Nonempty
                        hS : DirectedOn (fun x1 x2 => LE.le x1 x2) S
                        x : R
                        ⊢ Iff (Membership.mem (↑(SupSet.sSup S)) x) (Membership.mem (Set.iUnion fun s  …
                      -/
  Set.ext fun x => by simp [mem_sSup_of_directedOn Sne hS]
                      /-
                        🎉 no goals
                      -/


/-- Restriction of a ring homomorphism to a subsemiring of the codomain. -/
def codRestrict (f : R →+* S) (s : σS) (h : ∀ x, f x ∈ s) : R →+* s :=
  { (f : R →* S).codRestrict s h, (f : R →+ S).codRestrict s h with toFun := fun n => ⟨f n, h n⟩ }


@[simp]
theorem codRestrict_apply (f : R →+* S) (s : σS) (h : ∀ x, f x ∈ s) (x : R) :
    (f.codRestrict s h x : S) = f x :=
  rfl


/-- The ring homomorphism from the preimage of `s` to `s`. -/
def restrict (f : R →+* S) (s' : σR) (s : σS) (h : ∀ x ∈ s', f x ∈ s) : s' →+* s :=
  (f.domRestrict s').codRestrict s fun x => h x x.2


@[simp]
theorem coe_restrict_apply (f : R →+* S) (s' : σR) (s : σS) (h : ∀ x ∈ s', f x ∈ s) (x : s') :
    (f.restrict s' s h x : S) = f x :=
  rfl


@[simp]
theorem comp_restrict (f : R →+* S) (s' : σR) (s : σS) (h : ∀ x ∈ s', f x ∈ s) :
    (SubsemiringClass.subtype s).comp (f.restrict s' s h) = f.comp (SubsemiringClass.subtype s') :=
  rfl


/-- Restriction of a ring homomorphism to its range interpreted as a subsemiring.

This is the bundled version of `Set.rangeFactorization`. -/
def rangeSRestrict (f : R →+* S) : R →+* f.rangeS :=
  f.codRestrict (R := R) (S := S) (σS := Subsemiring S) f.rangeS f.mem_rangeS_self


@[simp]
theorem coe_rangeSRestrict (f : R →+* S) (x : R) : (f.rangeSRestrict x : S) = f x :=
  rfl


theorem rangeSRestrict_surjective (f : R →+* S) : Function.Surjective f.rangeSRestrict :=
  fun ⟨_, hy⟩ =>
  let ⟨x, hx⟩ := mem_rangeS.mp hy
  ⟨x, Subtype.ext hx⟩


theorem rangeS_top_iff_surjective {f : R →+* S} :
    f.rangeS = (⊤ : Subsemiring S) ↔ Function.Surjective f :=
                                          /-
                                            R : Type u
                                            S : Type v
                                            inst✝¹ : NonAssocSemiring R
                                            inst✝ : NonAssocSemiring S
                                            f : RingHom R S
                                            ⊢ Iff (Eq ↑f.rangeS ↑Top.top) (Eq (Set.range ⇑f) Set.univ)
                                          -/
  SetLike.ext'_iff.trans <| Iff.trans (by rw [coe_rangeS, coe_top]) Set.range_eq_univ
                                          /-
                                            🎉 no goals
                                          -/


/-- The range of a surjective ring homomorphism is the whole of the codomain. -/
@[simp]
theorem rangeS_top_of_surjective (f : R →+* S) (hf : Function.Surjective f) :
    f.rangeS = (⊤ : Subsemiring S) :=
  rangeS_top_iff_surjective.2 hf


/-- If two ring homomorphisms are equal on a set, then they are equal on its subsemiring closure. -/
theorem eqOn_sclosure {f g : R →+* S} {s : Set R} (h : Set.EqOn f g s) : Set.EqOn f g (closure s) :=
  show closure s ≤ f.eqLocusS g from closure_le.2 h


theorem eq_of_eqOn_stop {f g : R →+* S} (h : Set.EqOn f g (⊤ : Subsemiring R)) : f = g :=
  ext fun _ => h trivial


theorem eq_of_eqOn_sdense {s : Set R} (hs : closure s = ⊤) {f g : R →+* S} (h : s.EqOn f g) :
    f = g :=
  eq_of_eqOn_stop <| hs ▸ eqOn_sclosure h


theorem sclosure_preimage_le (f : R →+* S) (s : Set S) : closure (f ⁻¹' s) ≤ (closure s).comap f :=
  closure_le.2 fun _ hx => SetLike.mem_coe.2 <| mem_comap.2 <| subset_closure hx


/-- The image under a ring homomorphism of the subsemiring generated by a set equals
the subsemiring generated by the image of the set. -/
theorem map_closureS (f : R →+* S) (s : Set R) : (closure s).map f = closure (f '' s) :=
  Set.image_preimage.l_comm_of_u_comm (gc_map_comap f) (Subsemiring.gi S).gc (Subsemiring.gi R).gc
    fun _ ↦ rfl


/-- The ring homomorphism associated to an inclusion of subsemirings. -/
def inclusion {S T : Subsemiring R} (h : S ≤ T) : S →+* T :=
  S.subtype.codRestrict _ fun x => h x.2


@[simp]
theorem rangeS_subtype (s : Subsemiring R) : s.subtype.rangeS = s :=
  SetLike.coe_injective <| (coe_rangeS _).trans Subtype.range_coe


@[simp]
theorem range_fst : (fst R S).rangeS = ⊤ :=
  (fst R S).rangeS_top_of_surjective <| Prod.fst_surjective


@[simp]
theorem range_snd : (snd R S).rangeS = ⊤ :=
  (snd R S).rangeS_top_of_surjective <| Prod.snd_surjective


@[simp]
theorem prod_bot_sup_bot_prod (s : Subsemiring R) (t : Subsemiring S) :
    s.prod ⊥ ⊔ prod ⊥ t = s.prod t :=
  le_antisymm (sup_le (prod_mono_right s bot_le) (prod_mono_left t bot_le)) fun p hp =>
    Prod.fst_mul_snd p ▸
      mul_mem
        ((le_sup_left : s.prod ⊥ ≤ s.prod ⊥ ⊔ prod ⊥ t) ⟨hp.1, SetLike.mem_coe.2 <| one_mem ⊥⟩)
        ((le_sup_right : prod ⊥ t ≤ s.prod ⊥ ⊔ prod ⊥ t) ⟨SetLike.mem_coe.2 <| one_mem ⊥, hp.2⟩)


/-- Makes the identity isomorphism from a proof two subsemirings of a multiplicative
    monoid are equal. -/
def subsemiringCongr (h : s = t) : s ≃+* t :=
  {
    Equiv.setCongr <| congr_arg _ h with
    map_mul' := fun _ _ => rfl
    map_add' := fun _ _ => rfl }


/-- Restrict a ring homomorphism with a left inverse to a ring isomorphism to its
`RingHom.rangeS`. -/
def ofLeftInverseS {g : S → R} {f : R →+* S} (h : Function.LeftInverse g f) : R ≃+* f.rangeS :=
  { f.rangeSRestrict with
    toFun := fun x => f.rangeSRestrict x
    invFun := fun x => (g ∘ f.rangeS.subtype) x
    left_inv := h
    right_inv := fun x =>
      Subtype.ext <|
        let ⟨x', hx'⟩ := RingHom.mem_rangeS.mp x.prop
                            /-
                              R : Type u
                              S : Type v
                              T : Type w
                              inst✝² : NonAssocSemiring R
                              M : Submonoid R
                              inst✝¹ : NonAssocSemiring S
                              inst✝ : NonAssocSemiring T
                              s t : Subsemiring R
                              g : S → R
                              f : RingHom R S
                              h : Function.LeftInverse g ⇑f
                              x : Subtype fun x => Membership.mem f.rangeS x
                              x' : R
                              hx' : Eq (f x') ↑x
                              ⊢ Eq (f (g ↑x)) ↑x
                            -/
        show f (g x) = x by rw [← hx', h x'] }
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem ofLeftInverseS_apply {g : S → R} {f : R →+* S} (h : Function.LeftInverse g f) (x : R) :
    ↑(ofLeftInverseS h x) = f x :=
  rfl


@[simp]
theorem ofLeftInverseS_symm_apply {g : S → R} {f : R →+* S} (h : Function.LeftInverse g f)
    (x : f.rangeS) : (ofLeftInverseS h).symm x = g x :=
  rfl


/-- Given an equivalence `e : R ≃+* S` of semirings and a subsemiring `s` of `R`,
`subsemiring_map e s` is the induced equivalence between `s` and `s.map e` -/
@[simps!]
def subsemiringMap (e : R ≃+* S) (s : Subsemiring R) : s ≃+* s.map e.toRingHom :=
  { e.toAddEquiv.addSubmonoidMap s.toAddSubmonoid, e.toMulEquiv.submonoidMap s.toSubmonoid with }

-- These lemmas have always been bad (https://github.com/leanprover-community/mathlib4/issues/7657), but https://github.com/leanprover/lean4/pull/2644 made `simp` start noticing

/-- The action by a subsemiring is the action by the underlying semiring. -/
instance smul [SMul R' α] (S : Subsemiring R') : SMul S α :=
  S.toSubmonoid.smul


theorem smul_def [SMul R' α] {S : Subsemiring R'} (g : S) (m : α) : g • m = (g : R') • m :=
  rfl


instance smulCommClass_left [SMul R' β] [SMul α β] [SMulCommClass R' α β] (S : Subsemiring R') :
    SMulCommClass S α β :=
  S.toSubmonoid.smulCommClass_left


instance smulCommClass_right [SMul α β] [SMul R' β] [SMulCommClass α R' β] (S : Subsemiring R') :
    SMulCommClass α S β :=
  S.toSubmonoid.smulCommClass_right


/-- Note that this provides `IsScalarTower S R R` which is needed by `smul_mul_assoc`. -/
instance isScalarTower [SMul α β] [SMul R' α] [SMul R' β] [IsScalarTower R' α β]
    (S : Subsemiring R') :
    IsScalarTower S α β :=
  S.toSubmonoid.isScalarTower


instance faithfulSMul [SMul R' α] [FaithfulSMul R' α] (S : Subsemiring R') : FaithfulSMul S α :=
  S.toSubmonoid.faithfulSMul


/-- The action by a subsemiring is the action by the underlying semiring. -/
instance [Zero α] [SMulWithZero R' α] (S : Subsemiring R') : SMulWithZero S α :=
  SMulWithZero.compHom _ S.subtype.toMonoidWithZeroHom.toZeroHom


/-- The action by a subsemiring is the action by the underlying semiring. -/
instance mulAction [MulAction R' α] (S : Subsemiring R') : MulAction S α :=
  S.toSubmonoid.mulAction


/-- The action by a subsemiring is the action by the underlying semiring. -/
instance distribMulAction [AddMonoid α] [DistribMulAction R' α] (S : Subsemiring R') :
    DistribMulAction S α :=
  S.toSubmonoid.distribMulAction


/-- The action by a subsemiring is the action by the underlying semiring. -/
instance mulDistribMulAction [Monoid α] [MulDistribMulAction R' α] (S : Subsemiring R') :
    MulDistribMulAction S α :=
  S.toSubmonoid.mulDistribMulAction


/-- The action by a subsemiring is the action by the underlying semiring. -/
instance mulActionWithZero [Zero α] [MulActionWithZero R' α] (S : Subsemiring R') :
    MulActionWithZero S α :=
  MulActionWithZero.compHom _ S.subtype.toMonoidWithZeroHom

-- Porting note: instance named explicitly for use in `RingTheory/Subring/Basic`

/-- The action by a subsemiring is the action by the underlying semiring. -/
instance module [AddCommMonoid α] [Module R' α] (S : Subsemiring R') : Module S α :=
  -- Porting note: copying over the `smul` field causes a timeout
  -- { Module.compHom _ S.subtype with smul := (· • ·) }
  Module.compHom _ S.subtype


/-- The action by a subsemiring is the action by the underlying semiring. -/
instance [Semiring α] [MulSemiringAction R' α] (S : Subsemiring R') : MulSemiringAction S α :=
  S.toSubmonoid.mulSemiringAction


/-- The center of a semiring acts commutatively on that semiring. -/
instance center.smulCommClass_left : SMulCommClass (center R') R' R' :=
  Submonoid.center.smulCommClass_left


/-- The center of a semiring acts commutatively on that semiring. -/
instance center.smulCommClass_right : SMulCommClass R' (center R') R' :=
  Submonoid.center.smulCommClass_right


lemma closure_le_centralizer_centralizer (s : Set R') :
    closure s ≤ centralizer (centralizer s) :=
  closure_le.mpr Set.subset_centralizer_centralizer


/-- If all the elements of a set `s` commute, then `closure s` is a commutative semiring. -/
abbrev closureCommSemiringOfComm {s : Set R'} (hcomm : ∀ x ∈ s, ∀ y ∈ s, x * y = y * x) :
    CommSemiring (closure s) :=
  { (closure s).toSemiring with
    mul_comm := fun ⟨_, h₁⟩ ⟨_, h₂⟩ ↦
      have := closure_le_centralizer_centralizer s
      Subtype.ext <| Set.centralizer_centralizer_comm_of_comm hcomm _ (this h₁) _ (this h₂) }


theorem map_comap_eq (f : R →+* S) (t : Subsemiring S) : (t.comap f).map f = t ⊓ f.rangeS :=
  SetLike.coe_injective Set.image_preimage_eq_inter_range


theorem map_comap_eq_self
    {f : R →+* S} {t : Subsemiring S} (h : t ≤ f.rangeS) : (t.comap f).map f = t := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : NonAssocSemiring R
    inst✝ : NonAssocSemiring S
    f : RingHom R S
    t : Subsemiring S
    h : LE.le t f.rangeS
    ⊢ Eq (Subsemiring.map f (Subsemiring.comap f t)) t
  -/
  simpa only [inf_of_le_left h] using map_comap_eq f t
  /-
    🎉 no goals
  -/


theorem map_comap_eq_self_of_surjective
    {f : R →+* S} (hf : Function.Surjective f) (t : Subsemiring S) : (t.comap f).map f = t :=
                          /-
                            R : Type u
                            S : Type v
                            inst✝¹ : NonAssocSemiring R
                            inst✝ : NonAssocSemiring S
                            f : RingHom R S
                            hf : Function.Surjective ⇑f
                            t : Subsemiring S
                            ⊢ LE.le t f.rangeS
                          -/
  map_comap_eq_self <| by simp [hf]
                          /-
                            🎉 no goals
                          -/


theorem comap_map_eq_self_of_injective
    {f : R →+* S} (hf : Function.Injective f) (s : Subsemiring R) : (s.map f).comap f = s :=
  SetLike.coe_injective (Set.preimage_image_eq _ hf)


