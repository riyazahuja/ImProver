/-- Product of a list of elements in a subfield is in the subfield. -/
protected theorem list_prod_mem {l : List K} : (∀ x ∈ l, x ∈ s) → l.prod ∈ s :=
  list_prod_mem


/-- Sum of a list of elements in a subfield is in the subfield. -/
protected theorem list_sum_mem {l : List K} : (∀ x ∈ l, x ∈ s) → l.sum ∈ s :=
  list_sum_mem


/-- Sum of a multiset of elements in a `Subfield` is in the `Subfield`. -/
protected theorem multiset_sum_mem (m : Multiset K) : (∀ a ∈ m, a ∈ s) → m.sum ∈ s :=
  multiset_sum_mem m


/-- Sum of elements in a `Subfield` indexed by a `Finset` is in the `Subfield`. -/
protected theorem sum_mem {ι : Type*} {t : Finset ι} {f : ι → K} (h : ∀ c ∈ t, f c ∈ s) :
    (∑ i ∈ t, f i) ∈ s :=
  sum_mem h


/-- The subfield of `K` containing all elements of `K`. -/
instance : Top (Subfield K) :=
  ⟨{ (⊤ : Subring K) with inv_mem' := fun x _ => Subring.mem_top x }⟩


instance : Inhabited (Subfield K) :=
  ⟨⊤⟩


@[simp]
theorem mem_top (x : K) : x ∈ (⊤ : Subfield K) :=
  Set.mem_univ x


@[simp]
theorem coe_top : ((⊤ : Subfield K) : Set K) = Set.univ :=
  rfl


/-- The ring equiv between the top element of `Subfield K` and `K`. -/
def topEquiv : (⊤ : Subfield K) ≃+* K :=
  Subsemiring.topEquiv


/-- The preimage of a subfield along a ring homomorphism is a subfield. -/
def comap (s : Subfield L) : Subfield K :=
  { s.toSubring.comap f with
    inv_mem' := fun x hx =>
      show f x⁻¹ ∈ s by
        /-
          K : Type u
          L : Type v
          M : Type w
          inst✝² : DivisionRing K
          inst✝¹ : DivisionRing L
          inst✝ : DivisionRing M
          s✝ t : Subfield K
          f : RingHom K L
          s : Subfield L
          x : K
          hx : Membership.mem __src✝.carrier x
          ⊢ Membership.mem s (f (Inv.inv x))
        -/
        rw [map_inv₀ f]
        /-
          K : Type u
          L : Type v
          M : Type w
          inst✝² : DivisionRing K
          inst✝¹ : DivisionRing L
          inst✝ : DivisionRing M
          s✝ t : Subfield K
          f : RingHom K L
          s : Subfield L
          x : K
          hx : Membership.mem __src✝.carrier x
          ⊢ Membership.mem s (Inv.inv (f x))
        -/
        exact s.inv_mem hx }
        /-
          🎉 no goals
        -/


@[simp]
theorem coe_comap (s : Subfield L) : (s.comap f : Set K) = f ⁻¹' s :=
  rfl


@[simp]
theorem mem_comap {s : Subfield L} {f : K →+* L} {x : K} : x ∈ s.comap f ↔ f x ∈ s :=
  Iff.rfl


theorem comap_comap (s : Subfield M) (g : L →+* M) (f : K →+* L) :
    (s.comap g).comap f = s.comap (g.comp f) :=
  rfl


/-- The image of a subfield along a ring homomorphism is a subfield. -/
def map (s : Subfield K) : Subfield L :=
  { s.toSubring.map f with
    inv_mem' := by
      /-
        K : Type u
        L : Type v
        M : Type w
        inst✝² : DivisionRing K
        inst✝¹ : DivisionRing L
        inst✝ : DivisionRing M
        s✝ t : Subfield K
        f : RingHom K L
        s : Subfield K
        ⊢ ∀ (x : L), Membership.mem __src✝.carrier x → Membership.mem __src✝.carrier ( …
      -/
      rintro _ ⟨x, hx, rfl⟩
      /-
        case intro.intro
        K : Type u
        L : Type v
        M : Type w
        inst✝² : DivisionRing K
        inst✝¹ : DivisionRing L
        inst✝ : DivisionRing M
        s✝ t : Subfield K
        f : RingHom K L
        s : Subfield K
        x : K
        hx : Membership.mem s.carrier x
        ⊢ Membership.mem __src✝.carrier (Inv.inv (f x))
      -/
      exact ⟨x⁻¹, s.inv_mem hx, map_inv₀ f x⟩ }
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_map : (s.map f : Set L) = f '' s :=
  rfl


@[simp]
theorem mem_map {f : K →+* L} {s : Subfield K} {y : L} : y ∈ s.map f ↔ ∃ x ∈ s, f x = y := by
  /-
    K : Type u
    L : Type v
    inst✝¹ : DivisionRing K
    inst✝ : DivisionRing L
    f : RingHom K L
    s : Subfield K
    y : L
    ⊢ Iff (Membership.mem (Subfield.map f s) y) (Exists fun x => And (Membership.m …
  -/
  unfold map
  /-
    K : Type u
    L : Type v
    inst✝¹ : DivisionRing K
    inst✝ : DivisionRing L
    f : RingHom K L
    s : Subfield K
    y : L
    ⊢ Iff
        (Membership.mem
          (let __src := Subring.map f s.toSubring;
          { toSubring := __src, inv_mem' := ⋯ })
          y)
        (Exists fun x => And (Membership.mem s x) (Eq (f x) y))
  -/
  simp only [mem_mk, Subring.mem_mk, Subring.mem_toSubsemiring, Subring.mem_map, mem_toSubring]
  /-
    🎉 no goals
  -/


theorem map_map (g : L →+* M) (f : K →+* L) : (s.map f).map g = s.map (g.comp f) :=
  SetLike.ext' <| Set.image_image _ _ _


theorem map_le_iff_le_comap {f : K →+* L} {s : Subfield K} {t : Subfield L} :
    s.map f ≤ t ↔ s ≤ t.comap f :=
  Set.image_subset_iff


theorem gc_map_comap (f : K →+* L) : GaloisConnection (map f) (comap f) := fun _ _ =>
  map_le_iff_le_comap


/-- The range of a ring homomorphism, as a subfield of the target. See Note [range copy pattern]. -/
def fieldRange : Subfield L :=
  ((⊤ : Subfield K).map f).copy (Set.range f) Set.image_univ.symm


@[simp]
theorem coe_fieldRange : (f.fieldRange : Set L) = Set.range f :=
  rfl


@[simp]
theorem mem_fieldRange {f : K →+* L} {y : L} : y ∈ f.fieldRange ↔ ∃ x, f x = y :=
  Iff.rfl


theorem fieldRange_eq_map : f.fieldRange = Subfield.map f ⊤ := by
  /-
    K : Type u
    L : Type v
    inst✝¹ : DivisionRing K
    inst✝ : DivisionRing L
    f : RingHom K L
    ⊢ Eq f.fieldRange (Subfield.map f Top.top)
  -/
  ext
  /-
    case h
    K : Type u
    L : Type v
    inst✝¹ : DivisionRing K
    inst✝ : DivisionRing L
    f : RingHom K L
    x✝ : L
    ⊢ Iff (Membership.mem f.fieldRange x✝) (Membership.mem (Subfield.map f Top.top …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem map_fieldRange : f.fieldRange.map g = (g.comp f).fieldRange := by
  /-
    K : Type u
    L : Type v
    M : Type w
    inst✝² : DivisionRing K
    inst✝¹ : DivisionRing L
    inst✝ : DivisionRing M
    g : RingHom L M
    f : RingHom K L
    ⊢ Eq (Subfield.map g f.fieldRange) (g.comp f).fieldRange
  -/
  simpa only [fieldRange_eq_map] using (⊤ : Subfield K).map_map g f
  /-
    🎉 no goals
  -/


theorem mem_fieldRange_self (x : K) : f x ∈ f.fieldRange :=
  exists_apply_eq_apply _ _


theorem fieldRange_eq_top_iff {f : K →+* L} :
    f.fieldRange = ⊤ ↔ Function.Surjective f :=
  SetLike.ext'_iff.trans Set.range_eq_univ


/-- The range of a morphism of fields is a fintype, if the domain is a fintype.

Note that this instance can cause a diamond with `Subtype.Fintype` if `L` is also a fintype. -/
instance fintypeFieldRange [Fintype K] [DecidableEq L] (f : K →+* L) : Fintype f.fieldRange :=
  Set.fintypeRange f


/-- The inf of two subfields is their intersection. -/
instance : Min (Subfield K) :=
  ⟨fun s t =>
    { s.toSubring ⊓ t.toSubring with
      inv_mem' := fun _ hx =>
        Subring.mem_inf.mpr
          ⟨s.inv_mem (Subring.mem_inf.mp hx).1, t.inv_mem (Subring.mem_inf.mp hx).2⟩ }⟩


@[simp]
theorem coe_inf (p p' : Subfield K) : ((p ⊓ p' : Subfield K) : Set K) = p.carrier ∩ p'.carrier :=
  rfl


@[simp]
theorem mem_inf {p p' : Subfield K} {x : K} : x ∈ p ⊓ p' ↔ x ∈ p ∧ x ∈ p' :=
  Iff.rfl


instance : InfSet (Subfield K) :=
  ⟨fun S =>
    { sInf (Subfield.toSubring '' S) with
      inv_mem' := by
        /-
          K : Type u
          L : Type v
          M : Type w
          inst✝² : DivisionRing K
          inst✝¹ : DivisionRing L
          inst✝ : DivisionRing M
          S : Set (Subfield K)
          ⊢ ∀ (x : K), Membership.mem __src✝.carrier x → Membership.mem __src✝.carrier ( …
        -/
        rintro x hx
        /-
          K : Type u
          L : Type v
          M : Type w
          inst✝² : DivisionRing K
          inst✝¹ : DivisionRing L
          inst✝ : DivisionRing M
          S : Set (Subfield K)
          x : K
          hx : Membership.mem __src✝.carrier x
          ⊢ Membership.mem __src✝.carrier (Inv.inv x)
        -/
        apply Subring.mem_sInf.mpr
        /-
          K : Type u
          L : Type v
          M : Type w
          inst✝² : DivisionRing K
          inst✝¹ : DivisionRing L
          inst✝ : DivisionRing M
          S : Set (Subfield K)
          x : K
          hx : Membership.mem __src✝.carrier x
          ⊢ ∀ (p : Subring K), Membership.mem (Set.image Subfield.toSubring S) p → Membe …
        -/
        rintro _ ⟨p, p_mem, rfl⟩
        /-
          case intro.intro
          K : Type u
          L : Type v
          M : Type w
          inst✝² : DivisionRing K
          inst✝¹ : DivisionRing L
          inst✝ : DivisionRing M
          S : Set (Subfield K)
          x : K
          hx : Membership.mem __src✝.carrier x
          p : Subfield K
          p_mem : Membership.mem S p
          ⊢ Membership.mem p.toSubring (Inv.inv x)
        -/
        exact p.inv_mem (Subring.mem_sInf.mp hx p.toSubring ⟨p, p_mem, rfl⟩) }⟩
        /-
          🎉 no goals
        -/


@[simp, norm_cast]
theorem coe_sInf (S : Set (Subfield K)) : ((sInf S : Subfield K) : Set K) = ⋂ s ∈ S, ↑s :=
  show ((sInf (Subfield.toSubring '' S) : Subring K) : Set K) = ⋂ s ∈ S, ↑s by
    /-
      K : Type u
      inst✝ : DivisionRing K
      S : Set (Subfield K)
      ⊢ Eq (↑(InfSet.sInf (Set.image Subfield.toSubring S))) (Set.iInter fun s => Se …
    -/
    ext x
    /-
      case h
      K : Type u
      inst✝ : DivisionRing K
      S : Set (Subfield K)
      x : K
      ⊢ Iff (Membership.mem (↑(InfSet.sInf (Set.image Subfield.toSubring S))) x) (Me …
    -/
    rw [Subring.coe_sInf, Set.mem_iInter, Set.mem_iInter]
    exact
      ⟨fun h s s' ⟨s_mem, s'_eq⟩ => h s.toSubring _ ⟨⟨s, s_mem, rfl⟩, s'_eq⟩,
        fun h s s' ⟨⟨s'', s''_mem, s_eq⟩, (s'_eq : ↑s = s')⟩ =>
        h s'' _ ⟨s''_mem, by simp [← s_eq, ← s'_eq]⟩⟩


theorem mem_sInf {S : Set (Subfield K)} {x : K} : x ∈ sInf S ↔ ∀ p ∈ S, x ∈ p :=
  Subring.mem_sInf.trans
    ⟨fun h p hp => h p.toSubring ⟨p, hp, rfl⟩, fun h _ ⟨p', hp', p_eq⟩ => p_eq ▸ h p' hp'⟩


@[simp, norm_cast]
theorem coe_iInf {ι : Sort*} {S : ι → Subfield K} : (↑(⨅ i, S i) : Set K) = ⋂ i, S i := by
  /-
    K : Type u
    inst✝ : DivisionRing K
    ι : Sort u_1
    S : ι → Subfield K
    ⊢ Eq (↑(iInf fun i => S i)) (Set.iInter fun i => ↑(S i))
  -/
  simp only [iInf, coe_sInf, Set.biInter_range]
  /-
    🎉 no goals
  -/


theorem mem_iInf {ι : Sort*} {S : ι → Subfield K} {x : K} : (x ∈ ⨅ i, S i) ↔ ∀ i, x ∈ S i := by
  /-
    K : Type u
    inst✝ : DivisionRing K
    ι : Sort u_1
    S : ι → Subfield K
    x : K
    ⊢ Iff (Membership.mem (iInf fun i => S i) x) (∀ (i : ι), Membership.mem (S i) x)
  -/
  simp only [iInf, mem_sInf, Set.forall_mem_range]
  /-
    🎉 no goals
  -/


@[simp]
theorem sInf_toSubring (s : Set (Subfield K)) :
    (sInf s).toSubring = ⨅ t ∈ s, Subfield.toSubring t := by
  /-
    K : Type u
    inst✝ : DivisionRing K
    s : Set (Subfield K)
    ⊢ Eq (InfSet.sInf s).toSubring (iInf fun t => iInf fun h => t.toSubring)
  -/
  ext x
  /-
    case h
    K : Type u
    inst✝ : DivisionRing K
    s : Set (Subfield K)
    x : K
    ⊢ Iff (Membership.mem (InfSet.sInf s).toSubring x) (Membership.mem (iInf fun t …
  -/
  rw [mem_toSubring, mem_sInf]
  /-
    case h
    K : Type u
    inst✝ : DivisionRing K
    s : Set (Subfield K)
    x : K
    ⊢ Iff (∀ (p : Subfield K), Membership.mem s p → Membership.mem p x) (Membershi …
  -/
  erw [Subring.mem_sInf]
  exact
    ⟨fun h p ⟨p', hp⟩ => hp ▸ Subring.mem_sInf.mpr fun p ⟨hp', hp⟩ => hp ▸ h _ hp', fun h p hp =>
      h p.toSubring
        ⟨p,
          Subring.ext fun x =>
            ⟨fun hx => Subring.mem_sInf.mp hx _ ⟨hp, rfl⟩, fun hx =>
              Subring.mem_sInf.mpr fun p' ⟨_, p'_eq⟩ => p'_eq ▸ hx⟩⟩⟩


theorem isGLB_sInf (S : Set (Subfield K)) : IsGLB S (sInf S) := by
  /-
    K : Type u
    inst✝ : DivisionRing K
    S : Set (Subfield K)
    ⊢ IsGLB S (InfSet.sInf S)
  -/
  have : ∀ {s t : Subfield K}, (s : Set K) ≤ t ↔ s ≤ t := by simp [SetLike.coe_subset_coe]
  /-
    K : Type u
    inst✝ : DivisionRing K
    S : Set (Subfield K)
    this : ∀ {s t : Subfield K}, Iff (LE.le ↑s ↑t) (LE.le s t)
    ⊢ IsGLB S (InfSet.sInf S)
  -/
  refine IsGLB.of_image this ?_
  /-
    K : Type u
    inst✝ : DivisionRing K
    S : Set (Subfield K)
    this : ∀ {s t : Subfield K}, Iff (LE.le ↑s ↑t) (LE.le s t)
    ⊢ IsGLB (Set.image (fun {x} => ↑x) S) ↑(InfSet.sInf S)
  -/
  convert isGLB_biInf (s := S) (f := SetLike.coe)
  /-
    case h.e'_4
    K : Type u
    inst✝ : DivisionRing K
    S : Set (Subfield K)
    this : ∀ {s t : Subfield K}, Iff (LE.le ↑s ↑t) (LE.le s t)
    ⊢ Eq (↑(InfSet.sInf S)) (iInf fun x => iInf fun h => ↑x)
  -/
  exact coe_sInf _
  /-
    🎉 no goals
  -/


/-- Subfields of a ring form a complete lattice. -/
instance : CompleteLattice (Subfield K) :=
  { completeLatticeOfInf (Subfield K) isGLB_sInf with
    top := ⊤
    le_top := fun _ _ _ => trivial
    inf := (· ⊓ ·)
    inf_le_left := fun _ _ _ => And.left
    inf_le_right := fun _ _ _ => And.right
    le_inf := fun _ _ _ h₁ h₂ _ hx => ⟨h₁ hx, h₂ hx⟩ }


/-- The `Subfield` generated by a set. -/
def closure (s : Set K) : Subfield K := sInf {S | s ⊆ S}


theorem mem_closure {x : K} {s : Set K} : x ∈ closure s ↔ ∀ S : Subfield K, s ⊆ S → x ∈ S :=
  mem_sInf


/-- The subfield generated by a set includes the set. -/
@[simp, aesop safe 20 apply (rule_sets := [SetLike])]
theorem subset_closure {s : Set K} : s ⊆ closure s := fun _ hx => mem_closure.2 fun _ hS => hS hx


theorem subring_closure_le (s : Set K) : Subring.closure s ≤ (closure s).toSubring :=
  Subring.closure_le.mpr subset_closure


theorem not_mem_of_not_mem_closure {s : Set K} {P : K} (hP : P ∉ closure s) : P ∉ s := fun h =>
  hP (subset_closure h)


/-- A subfield `t` includes `closure s` if and only if it includes `s`. -/
@[simp]
theorem closure_le {s : Set K} {t : Subfield K} : closure s ≤ t ↔ s ⊆ t :=
  ⟨Set.Subset.trans subset_closure, fun h _ hx => mem_closure.mp hx t h⟩


/-- Subfield closure of a set is monotone in its argument: if `s ⊆ t`,
then `closure s ≤ closure t`. -/
@[gcongr]
theorem closure_mono ⦃s t : Set K⦄ (h : s ⊆ t) : closure s ≤ closure t :=
  closure_le.2 <| Set.Subset.trans h subset_closure


theorem closure_eq_of_le {s : Set K} {t : Subfield K} (h₁ : s ⊆ t) (h₂ : t ≤ closure s) :
    closure s = t :=
  le_antisymm (closure_le.2 h₁) h₂


/-- An induction principle for closure membership. If `p` holds for `1`, and all elements
of `s`, and is preserved under addition, negation, and multiplication, then `p` holds for all
elements of the closure of `s`. -/
@[elab_as_elim]
theorem closure_induction {s : Set K} {p : ∀ x ∈ closure s, Prop}
    (mem : ∀ x hx, p x (subset_closure hx))
    (one : p 1 (one_mem _)) (add : ∀ x y hx hy, p x hx → p y hy → p (x + y) (add_mem hx hy))
    (neg : ∀ x hx, p x hx → p (-x) (neg_mem hx)) (inv : ∀ x hx, p x hx → p x⁻¹ (inv_mem hx))
    (mul : ∀ x y hx hy, p x hx → p y hy → p (x * y) (mul_mem hx hy))
    {x} (h : x ∈ closure s) : p x h :=
  letI : Subfield K :=
    { carrier := {x | ∃ hx, p x hx}
                     /-
                       K : Type u
                       inst✝ : DivisionRing K
                       s : Set K
                       p : (x : K) → Membership.mem (Subfield.closure s) x → Prop
                       mem : ∀ (x : K) (hx : Membership.mem s x), p x ⋯
                       one : p 1 ⋯
                       add : ∀ (x y : K) (hx : Membership.mem (Subfield.closure s) x) (hy : Membershi …
                       neg : ∀ (x : K) (hx : Membership.mem (Subfield.closure s) x), p x hx → p (Neg. …
                       inv : ∀ (x : K) (hx : Membership.mem (Subfield.closure s) x), p x hx → p (Inv. …
                       mul : ∀ (x y : K) (hx : Membership.mem (Subfield.closure s) x) (hy : Membershi …
                       x : K
                       h : Membership.mem (Subfield.closure s) x
                       ⊢ ∀ {a b : K}, Membership.mem (setOf fun x => Exists fun hx => p x hx) a → Mem …
                     -/
      mul_mem' := by rintro _ _ ⟨_, hx⟩ ⟨_, hy⟩; exact ⟨_, mul _ _ _ _ hx hy⟩
                                                 /-
                                                   🎉 no goals
                                                 -/
      one_mem' := ⟨_, one⟩
                     /-
                       K : Type u
                       inst✝ : DivisionRing K
                       s : Set K
                       p : (x : K) → Membership.mem (Subfield.closure s) x → Prop
                       mem : ∀ (x : K) (hx : Membership.mem s x), p x ⋯
                       one : p 1 ⋯
                       add : ∀ (x y : K) (hx : Membership.mem (Subfield.closure s) x) (hy : Membershi …
                       neg : ∀ (x : K) (hx : Membership.mem (Subfield.closure s) x), p x hx → p (Neg. …
                       inv : ∀ (x : K) (hx : Membership.mem (Subfield.closure s) x), p x hx → p (Inv. …
                       mul : ∀ (x y : K) (hx : Membership.mem (Subfield.closure s) x) (hy : Membershi …
                       x : K
                       h : Membership.mem (Subfield.closure s) x
                       ⊢ ∀ {a b : K}, Membership.mem { carrier := setOf fun x => Exists fun hx => p x …
                     -/
      add_mem' := by rintro _ _ ⟨_, hx⟩ ⟨_, hy⟩; exact ⟨_, add _ _ _ _ hx hy⟩
                                                 /-
                                                   🎉 no goals
                                                 -/
      zero_mem' := ⟨zero_mem _, by
        /-
          K : Type u
          inst✝ : DivisionRing K
          s : Set K
          p : (x : K) → Membership.mem (Subfield.closure s) x → Prop
          mem : ∀ (x : K) (hx : Membership.mem s x), p x ⋯
          one : p 1 ⋯
          add : ∀ (x y : K) (hx : Membership.mem (Subfield.closure s) x) (hy : Membershi …
          neg : ∀ (x : K) (hx : Membership.mem (Subfield.closure s) x), p x hx → p (Neg. …
          inv : ∀ (x : K) (hx : Membership.mem (Subfield.closure s) x), p x hx → p (Inv. …
          mul : ∀ (x y : K) (hx : Membership.mem (Subfield.closure s) x) (hy : Membershi …
          x : K
          h : Membership.mem (Subfield.closure s) x
          ⊢ p 0 ⋯
        -/
        simp_rw [← @add_neg_cancel K _ 1]; exact add _ _ _ _ one (neg _ _ one)⟩
                                           /-
                                             🎉 no goals
                                           -/
                     /-
                       K : Type u
                       inst✝ : DivisionRing K
                       s : Set K
                       p : (x : K) → Membership.mem (Subfield.closure s) x → Prop
                       mem : ∀ (x : K) (hx : Membership.mem s x), p x ⋯
                       one : p 1 ⋯
                       add : ∀ (x y : K) (hx : Membership.mem (Subfield.closure s) x) (hy : Membershi …
                       neg : ∀ (x : K) (hx : Membership.mem (Subfield.closure s) x), p x hx → p (Neg. …
                       inv : ∀ (x : K) (hx : Membership.mem (Subfield.closure s) x), p x hx → p (Inv. …
                       mul : ∀ (x y : K) (hx : Membership.mem (Subfield.closure s) x) (hy : Membershi …
                       x : K
                       h : Membership.mem (Subfield.closure s) x
                       ⊢ ∀ {x : K}, Membership.mem { carrier := setOf fun x => Exists fun hx => p x h …
                     -/
      neg_mem' := by rintro _ ⟨_, hx⟩; exact ⟨_, neg _ _ hx⟩
                                       /-
                                         🎉 no goals
                                       -/
                     /-
                       K : Type u
                       inst✝ : DivisionRing K
                       s : Set K
                       p : (x : K) → Membership.mem (Subfield.closure s) x → Prop
                       mem : ∀ (x : K) (hx : Membership.mem s x), p x ⋯
                       one : p 1 ⋯
                       add : ∀ (x y : K) (hx : Membership.mem (Subfield.closure s) x) (hy : Membershi …
                       neg : ∀ (x : K) (hx : Membership.mem (Subfield.closure s) x), p x hx → p (Neg. …
                       inv : ∀ (x : K) (hx : Membership.mem (Subfield.closure s) x), p x hx → p (Inv. …
                       mul : ∀ (x y : K) (hx : Membership.mem (Subfield.closure s) x) (hy : Membershi …
                       x : K
                       h : Membership.mem (Subfield.closure s) x
                       ⊢ ∀ (x : K), Membership.mem { carrier := setOf fun x => Exists fun hx => p x h …
                     -/
      inv_mem' := by rintro _ ⟨_, hx⟩; exact ⟨_, inv _ _ hx⟩ }
                                       /-
                                         🎉 no goals
                                       -/
  ((closure_le (t := this)).2 (fun x hx ↦ ⟨_, mem x hx⟩) h).2


/-- `closure` forms a Galois insertion with the coercion to set. -/
protected def gi : GaloisInsertion (@closure K _) (↑) where
  choice s _ := closure s
  gc _ _ := closure_le
  le_l_u _ := subset_closure
  choice_eq _ _ := rfl


/-- Closure of a subfield `S` equals `S`. -/
theorem closure_eq (s : Subfield K) : closure (s : Set K) = s :=
  (Subfield.gi K).l_u_eq s


@[simp]
theorem closure_empty : closure (∅ : Set K) = ⊥ :=
  (Subfield.gi K).gc.l_bot


@[simp]
theorem closure_univ : closure (Set.univ : Set K) = ⊤ :=
  @coe_top K _ ▸ closure_eq ⊤


theorem closure_union (s t : Set K) : closure (s ∪ t) = closure s ⊔ closure t :=
  (Subfield.gi K).gc.l_sup


theorem closure_iUnion {ι} (s : ι → Set K) : closure (⋃ i, s i) = ⨆ i, closure (s i) :=
  (Subfield.gi K).gc.l_iSup


theorem closure_sUnion (s : Set (Set K)) : closure (⋃₀ s) = ⨆ t ∈ s, closure t :=
  (Subfield.gi K).gc.l_sSup


theorem map_sup (s t : Subfield K) (f : K →+* L) : (s ⊔ t).map f = s.map f ⊔ t.map f :=
  (gc_map_comap f).l_sup


theorem map_iSup {ι : Sort*} (f : K →+* L) (s : ι → Subfield K) :
    (iSup s).map f = ⨆ i, (s i).map f :=
  (gc_map_comap f).l_iSup


theorem map_inf (s t : Subfield K) (f : K →+* L) : (s ⊓ t).map f = s.map f ⊓ t.map f :=
  SetLike.coe_injective (Set.image_inter f.injective)


theorem map_iInf {ι : Sort*} [Nonempty ι] (f : K →+* L) (s : ι → Subfield K) :
    (iInf s).map f = ⨅ i, (s i).map f := by
  /-
    K : Type u
    L : Type v
    inst✝² : DivisionRing K
    inst✝¹ : DivisionRing L
    ι : Sort u_1
    inst✝ : Nonempty ι
    f : RingHom K L
    s : ι → Subfield K
    ⊢ Eq (Subfield.map f (iInf s)) (iInf fun i => Subfield.map f (s i))
  -/
  apply SetLike.coe_injective
  /-
    case a
    K : Type u
    L : Type v
    inst✝² : DivisionRing K
    inst✝¹ : DivisionRing L
    ι : Sort u_1
    inst✝ : Nonempty ι
    f : RingHom K L
    s : ι → Subfield K
    ⊢ Eq ↑(Subfield.map f (iInf s)) ↑(iInf fun i => Subfield.map f (s i))
  -/
  simpa using (Set.injOn_of_injective f.injective).image_iInter_eq (s := SetLike.coe ∘ s)
  /-
    🎉 no goals
  -/


theorem comap_inf (s t : Subfield L) (f : K →+* L) : (s ⊓ t).comap f = s.comap f ⊓ t.comap f :=
  (gc_map_comap f).u_inf


theorem comap_iInf {ι : Sort*} (f : K →+* L) (s : ι → Subfield L) :
    (iInf s).comap f = ⨅ i, (s i).comap f :=
  (gc_map_comap f).u_iInf


@[simp]
theorem map_bot (f : K →+* L) : (⊥ : Subfield K).map f = ⊥ :=
  (gc_map_comap f).l_bot


@[simp]
theorem comap_top (f : K →+* L) : (⊤ : Subfield L).comap f = ⊤ :=
  (gc_map_comap f).u_top


/-- The underlying set of a non-empty directed sSup of subfields is just a union of the subfields.
  Note that this fails without the directedness assumption (the union of two subfields is
  typically not a subfield) -/
theorem mem_iSup_of_directed {ι} [hι : Nonempty ι] {S : ι → Subfield K} (hS : Directed (· ≤ ·) S)
    {x : K} : (x ∈ ⨆ i, S i) ↔ ∃ i, x ∈ S i := by
  let s : Subfield K :=
    { __ := Subring.copy _ _ (Subring.coe_iSup_of_directed hS).symm
      inv_mem' := fun _ hx ↦ have ⟨i, hi⟩ := Set.mem_iUnion.mp hx
        Set.mem_iUnion.mpr ⟨i, (S i).inv_mem hi⟩ }
  have : iSup S = s := le_antisymm
    (iSup_le fun i ↦ le_iSup (fun i ↦ (S i : Set K)) i) (Set.iUnion_subset fun _ ↦ le_iSup S _)
  /-
    K : Type u
    inst✝ : DivisionRing K
    ι : Sort u_1
    hι : Nonempty ι
    S : ι → Subfield K
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x : K
    s : Subfield K :=
      let __spread.0 := (iSup fun i => (S i).toSubring).copy (Set.iUnion fun i =>  …
      { toSubring := __spread.0, inv_mem' := ⋯ }
    this : Eq (iSup S) s
    ⊢ Iff (Membership.mem (iSup fun i => S i) x) (Exists fun i => Membership.mem ( …
  -/
  exact this ▸ Set.mem_iUnion
  /-
    🎉 no goals
  -/


theorem coe_iSup_of_directed {ι} [hι : Nonempty ι] {S : ι → Subfield K} (hS : Directed (· ≤ ·) S) :
    ((⨆ i, S i : Subfield K) : Set K) = ⋃ i, ↑(S i) :=
                      /-
                        K : Type u
                        inst✝ : DivisionRing K
                        ι : Sort u_1
                        hι : Nonempty ι
                        S : ι → Subfield K
                        hS : Directed (fun x1 x2 => LE.le x1 x2) S
                        x : K
                        ⊢ Iff (Membership.mem (↑(iSup fun i => S i)) x) (Membership.mem (Set.iUnion fu …
                      -/
  Set.ext fun x => by simp [mem_iSup_of_directed hS]
                      /-
                        🎉 no goals
                      -/


theorem mem_sSup_of_directedOn {S : Set (Subfield K)} (Sne : S.Nonempty) (hS : DirectedOn (· ≤ ·) S)
    {x : K} : x ∈ sSup S ↔ ∃ s ∈ S, x ∈ s := by
  /-
    K : Type u
    inst✝ : DivisionRing K
    S : Set (Subfield K)
    Sne : S.Nonempty
    hS : DirectedOn (fun x1 x2 => LE.le x1 x2) S
    x : K
    ⊢ Iff (Membership.mem (SupSet.sSup S) x) (Exists fun s => And (Membership.mem  …
  -/
  haveI : Nonempty S := Sne.to_subtype
  /-
    K : Type u
    inst✝ : DivisionRing K
    S : Set (Subfield K)
    Sne : S.Nonempty
    hS : DirectedOn (fun x1 x2 => LE.le x1 x2) S
    x : K
    this : Nonempty ↑S
    ⊢ Iff (Membership.mem (SupSet.sSup S) x) (Exists fun s => And (Membership.mem  …
  -/
  simp only [sSup_eq_iSup', mem_iSup_of_directed hS.directed_val, Subtype.exists, exists_prop]
  /-
    🎉 no goals
  -/


theorem coe_sSup_of_directedOn {S : Set (Subfield K)} (Sne : S.Nonempty)
    (hS : DirectedOn (· ≤ ·) S) : (↑(sSup S) : Set K) = ⋃ s ∈ S, ↑s :=
                      /-
                        K : Type u
                        inst✝ : DivisionRing K
                        S : Set (Subfield K)
                        Sne : S.Nonempty
                        hS : DirectedOn (fun x1 x2 => LE.le x1 x2) S
                        x : K
                        ⊢ Iff (Membership.mem (↑(SupSet.sSup S)) x) (Membership.mem (Set.iUnion fun s  …
                      -/
  Set.ext fun x => by simp [mem_sSup_of_directedOn Sne hS]
                      /-
                        🎉 no goals
                      -/


/-- Restriction of a ring homomorphism to its range interpreted as a subfield. -/
def rangeRestrictField (f : K →+* L) : K →+* f.fieldRange :=
  f.rangeSRestrict


@[simp]
theorem coe_rangeRestrictField (f : K →+* L) (x : K) : (f.rangeRestrictField x : L) = f x :=
  rfl


/-- The subfield of elements `x : R` such that `f x = g x`, i.e.,
the equalizer of f and g as a subfield of R -/
def eqLocusField (f g : K →+* L) : Subfield K where
  __ := (f : K →+* L).eqLocus g
  inv_mem' _ := eq_on_inv₀ f g
  carrier := { x | f x = g x }


/-- If two ring homomorphisms are equal on a set, then they are equal on its subfield closure. -/
theorem eqOn_field_closure {f g : K →+* L} {s : Set K} (h : Set.EqOn f g s) :
    Set.EqOn f g (closure s) :=
  show closure s ≤ f.eqLocusField g from closure_le.2 h


theorem eq_of_eqOn_subfield_top {f g : K →+* L} (h : Set.EqOn f g (⊤ : Subfield K)) : f = g :=
  ext fun _ => h trivial


theorem eq_of_eqOn_of_field_closure_eq_top {s : Set K} (hs : closure s = ⊤) {f g : K →+* L}
    (h : s.EqOn f g) : f = g :=
  eq_of_eqOn_subfield_top <| hs ▸ eqOn_field_closure h


theorem field_closure_preimage_le (f : K →+* L) (s : Set L) :
    closure (f ⁻¹' s) ≤ (closure s).comap f :=
  closure_le.2 fun _ hx => SetLike.mem_coe.2 <| mem_comap.2 <| subset_closure hx


/-- The image under a ring homomorphism of the subfield generated by a set equals
the subfield generated by the image of the set. -/
theorem map_field_closure (f : K →+* L) (s : Set K) : (closure s).map f = closure (f '' s) :=
  Set.image_preimage.l_comm_of_u_comm (gc_map_comap f) (Subfield.gi L).gc (Subfield.gi K).gc
    fun _ ↦ rfl


/-- The ring homomorphism associated to an inclusion of subfields. -/
def inclusion {S T : Subfield K} (h : S ≤ T) : S →+* T :=
  S.subtype.codRestrict _ fun x => h x.2


@[simp]
theorem fieldRange_subtype (s : Subfield K) : s.subtype.fieldRange = s :=
  SetLike.ext' <| (coe_rangeS _).trans Subtype.range_coe


/-- Makes the identity isomorphism from a proof two subfields of a multiplicative
    monoid are equal. -/
def subfieldCongr (h : s = t) : s ≃+* t :=
  { Equiv.setCongr <| SetLike.ext'_iff.1 h with
    map_mul' := fun _ _ => rfl
    map_add' := fun _ _ => rfl }


theorem closure_preimage_le (f : K →+* L) (s : Set L) : closure (f ⁻¹' s) ≤ (closure s).comap f :=
  closure_le.2 fun _ hx => SetLike.mem_coe.2 <| mem_comap.2 <| subset_closure hx


/-- Product of a multiset of elements in a subfield is in the subfield. -/
protected theorem multiset_prod_mem (m : Multiset K) : (∀ a ∈ m, a ∈ s) → m.prod ∈ s :=
  multiset_prod_mem m


/-- Product of elements of a subfield indexed by a `Finset` is in the subfield. -/
protected theorem prod_mem {ι : Type*} {t : Finset ι} {f : ι → K} (h : ∀ c ∈ t, f c ∈ s) :
    (∏ i ∈ t, f i) ∈ s :=
  prod_mem h


instance toAlgebra : Algebra s K :=
  RingHom.toAlgebra s.subtype


/-- The `Subfield` generated by a set in a field. -/
private def commClosure (s : Set K) : Subfield K where
  carrier := {z : K | ∃ x ∈ Subring.closure s, ∃ y ∈ Subring.closure s, x / y = z}
  zero_mem' := ⟨0, Subring.zero_mem _, 1, Subring.one_mem _, div_one _⟩
  one_mem' := ⟨1, Subring.one_mem _, 1, Subring.one_mem _, div_one _⟩
  neg_mem' {x} := by
    /-
      K✝ : Type u
      L : Type v
      M : Type w
      inst✝³ : DivisionRing K✝
      inst✝² : DivisionRing L
      inst✝¹ : DivisionRing M
      s✝¹ : Set K✝
      K : Type u
      inst✝ : Field K
      s✝ : Subfield K
      s : Set K
      x : K
      ⊢ Membership.mem { carrier := setOf fun z => Exists fun x => And (Membership.m …
    -/
    rintro ⟨y, hy, z, hz, x_eq⟩
    /-
      case intro.intro.intro.intro
      K✝ : Type u
      L : Type v
      M : Type w
      inst✝³ : DivisionRing K✝
      inst✝² : DivisionRing L
      inst✝¹ : DivisionRing M
      s✝¹ : Set K✝
      K : Type u
      inst✝ : Field K
      s✝ : Subfield K
      s : Set K
      x y : K
      hy : Membership.mem (Subring.closure s) y
      z : K
      hz : Membership.mem (Subring.closure s) z
      x_eq : Eq (HDiv.hDiv y z) x
      ⊢ Membership.mem { carrier := setOf fun z => Exists fun x => And (Membership.m …
    -/
    exact ⟨-y, Subring.neg_mem _ hy, z, hz, x_eq ▸ neg_div _ _⟩
    /-
      🎉 no goals
    -/
    /-
      K✝ : Type u
      L : Type v
      M : Type w
      inst✝³ : DivisionRing K✝
      inst✝² : DivisionRing L
      inst✝¹ : DivisionRing M
      s✝¹ : Set K✝
      K : Type u
      inst✝ : Field K
      s✝ : Subfield K
      s : Set K
      a✝ b✝ : K
      x_mem : Membership.mem { carrier := setOf fun z => Exists fun x => And (Member …
      y_mem : Membership.mem { carrier := setOf fun z => Exists fun x => And (Member …
      ⊢ Membership.mem { carrier := setOf fun z => Exists fun x => And (Membership.m …
    -/
                   /-
                     K✝ : Type u
                     L : Type v
                     M : Type w
                     inst✝³ : DivisionRing K✝
                     inst✝² : DivisionRing L
                     inst✝¹ : DivisionRing M
                     s✝¹ : Set K✝
                     K : Type u
                     inst✝ : Field K
                     s✝ : Subfield K
                     s : Set K
                     x : K
                     ⊢ Membership.mem { carrier := setOf fun z => Exists fun x => And (Membership.m …
                   -/
    /-
      case intro.intro.intro.intro
      K✝ : Type u
      L : Type v
      M : Type w
      inst✝³ : DivisionRing K✝
      inst✝² : DivisionRing L
      inst✝¹ : DivisionRing M
      s✝¹ : Set K✝
      K : Type u
      inst✝ : Field K
      s✝ : Subfield K
      s : Set K
      b✝ : K
      y_mem : Membership.mem { carrier := setOf fun z => Exists fun x => And (Member …
      nx : K
      hnx : Membership.mem (Subring.closure s) nx
      dx : K
      hdx : Membership.mem (Subring.closure s) dx
      x_mem : Membership.mem { carrier := setOf fun z => Exists fun x => And (Member …
      ⊢ Membership.mem { carrier := setOf fun z => Exists fun x => And (Membership.m …
    -/
  inv_mem' x := by rintro ⟨y, hy, z, hz, x_eq⟩; exact ⟨z, hz, y, hy, x_eq ▸ (inv_div _ _).symm⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      K✝ : Type u
      L : Type v
      M : Type w
      inst✝³ : DivisionRing K✝
      inst✝² : DivisionRing L
      inst✝¹ : DivisionRing M
      s✝¹ : Set K✝
      K : Type u
      inst✝ : Field K
      s✝ : Subfield K
      s : Set K
      nx : K
      hnx : Membership.mem (Subring.closure s) nx
      dx : K
      hdx : Membership.mem (Subring.closure s) dx
      x_mem : Membership.mem { carrier := setOf fun z => Exists fun x => And (Member …
      ny : K
      hny : Membership.mem (Subring.closure s) ny
      dy : K
      hdy : Membership.mem (Subring.closure s) dy
      y_mem : Membership.mem { carrier := setOf fun z => Exists fun x => And (Member …
      ⊢ Membership.mem { carrier := setOf fun z => Exists fun x => And (Membership.m …
    -/
                                                /-
                                                  🎉 no goals
                                                -/
                             /-
                               🎉 no goals
                             -/
    /-
      case neg
      K✝ : Type u
      L : Type v
      M : Type w
      inst✝³ : DivisionRing K✝
      inst✝² : DivisionRing L
      inst✝¹ : DivisionRing M
      s✝¹ : Set K✝
      K : Type u
      inst✝ : Field K
      s✝ : Subfield K
      s : Set K
      nx : K
      hnx : Membership.mem (Subring.closure s) nx
      dx : K
      hdx : Membership.mem (Subring.closure s) dx
      x_mem : Membership.mem { carrier := setOf fun z => Exists fun x => And (Member …
      ny : K
      hny : Membership.mem (Subring.closure s) ny
      dy : K
      hdy : Membership.mem (Subring.closure s) dy
      y_mem : Membership.mem { carrier := setOf fun z => Exists fun x => And (Member …
      hx0 : Not (Eq dx 0)
      ⊢ Membership.mem { carrier := setOf fun z => Exists fun x => And (Membership.m …
    -/
    /-
      K✝ : Type u
      L : Type v
      M : Type w
      inst✝³ : DivisionRing K✝
      inst✝² : DivisionRing L
      inst✝¹ : DivisionRing M
      s✝¹ : Set K✝
      K : Type u
      inst✝ : Field K
      s✝ : Subfield K
      s : Set K
      ⊢ ∀ {a b : K}, Membership.mem (setOf fun z => Exists fun x => And (Membership. …
    -/
  add_mem' x_mem y_mem := by
                             /-
                               🎉 no goals
                             -/
    -- Use `id` in the next 2 `obtain`s so that assumptions stay there for the `rwa`s below
    obtain ⟨nx, hnx, dx, hdx, rfl⟩ := id x_mem
    obtain ⟨ny, hny, dy, hdy, rfl⟩ := id y_mem
    by_cases hx0 : dx = 0; · rwa [hx0, div_zero, zero_add]
    by_cases hy0 : dy = 0; · rwa [hy0, div_zero, add_zero]
    exact
      ⟨nx * dy + dx * ny, Subring.add_mem _ (Subring.mul_mem _ hnx hdy) (Subring.mul_mem _ hdx hny),
        dx * dy, Subring.mul_mem _ hdx hdy, (div_add_div nx ny hx0 hy0).symm⟩
  mul_mem' := by
    rintro _ _ ⟨nx, hnx, dx, hdx, rfl⟩ ⟨ny, hny, dy, hdy, rfl⟩
    exact ⟨nx * ny, Subring.mul_mem _ hnx hny, dx * dy, Subring.mul_mem _ hdx hdy,
      (div_mul_div_comm _ _ _ _).symm⟩


private theorem commClosure_eq_closure {s : Set K} : commClosure s = closure s :=
  le_antisymm
    (fun _ ⟨_, hy, _, hz, eq⟩ ↦ eq ▸ div_mem (subring_closure_le s hy) (subring_closure_le s hz))
    (closure_le.mpr fun x hx ↦ ⟨x, Subring.subset_closure hx, 1, Subring.one_mem _, div_one x⟩)


theorem mem_closure_iff {s : Set K} {x} :
    x ∈ closure s ↔ ∃ y ∈ Subring.closure s, ∃ z ∈ Subring.closure s, y / z = x := by
  /-
    K : Type u
    inst✝ : Field K
    s : Set K
    x : K
    ⊢ Iff (Membership.mem (Subfield.closure s) x) (Exists fun y => And (Membership …
  -/
  rw [← commClosure_eq_closure]; rfl
                                 /-
                                   🎉 no goals
                                 -/


theorem map_comap_eq (f : K →+* L) (s : Subfield L) : (s.comap f).map f = s ⊓ f.fieldRange :=
  SetLike.coe_injective Set.image_preimage_eq_inter_range


theorem map_comap_eq_self
    {f : K →+* L} {s : Subfield L} (h : s ≤ f.fieldRange) : (s.comap f).map f = s := by
  /-
    K : Type u
    L : Type v
    inst✝¹ : DivisionRing K
    inst✝ : DivisionRing L
    f : RingHom K L
    s : Subfield L
    h : LE.le s f.fieldRange
    ⊢ Eq (Subfield.map f (Subfield.comap f s)) s
  -/
  simpa only [inf_of_le_left h] using map_comap_eq f s
  /-
    🎉 no goals
  -/


theorem map_comap_eq_self_of_surjective
    {f : K →+* L} (hf : Function.Surjective f) (s : Subfield L) : (s.comap f).map f = s :=
  SetLike.coe_injective (Set.image_preimage_eq _ hf)


theorem comap_map (f : K →+* L) (s : Subfield K) : (s.map f).comap f = s :=
  SetLike.coe_injective (Set.preimage_image_eq _ f.injective)


/-- The action by a subfield is the action by the underlying field. -/
instance [SMul K X] (F : Subfield K) : SMul F X :=
  inferInstanceAs (SMul F.toSubsemiring X)


theorem smul_def [SMul K X] {F : Subfield K} (g : F) (m : X) : g • m = (g : K) • m :=
  rfl


instance smulCommClass_left [SMul K Y] [SMul X Y] [SMulCommClass K X Y] (F : Subfield K) :
    SMulCommClass F X Y :=
  inferInstanceAs (SMulCommClass F.toSubsemiring X Y)


instance smulCommClass_right [SMul X Y] [SMul K Y] [SMulCommClass X K Y] (F : Subfield K) :
    SMulCommClass X F Y :=
  inferInstanceAs (SMulCommClass X F.toSubsemiring Y)


/-- Note that this provides `IsScalarTower F K K` which is needed by `smul_mul_assoc`. -/
instance [SMul X Y] [SMul K X] [SMul K Y] [IsScalarTower K X Y] (F : Subfield K) :
    IsScalarTower F X Y :=
  inferInstanceAs (IsScalarTower F.toSubsemiring X Y)


instance [SMul K X] [FaithfulSMul K X] (F : Subfield K) : FaithfulSMul F X :=
  inferInstanceAs (FaithfulSMul F.toSubsemiring X)


/-- The action by a subfield is the action by the underlying field. -/
instance [MulAction K X] (F : Subfield K) : MulAction F X :=
  inferInstanceAs (MulAction F.toSubsemiring X)


/-- The action by a subfield is the action by the underlying field. -/
instance [AddMonoid X] [DistribMulAction K X] (F : Subfield K) : DistribMulAction F X :=
  inferInstanceAs (DistribMulAction F.toSubsemiring X)


/-- The action by a subfield is the action by the underlying field. -/
instance [Monoid X] [MulDistribMulAction K X] (F : Subfield K) : MulDistribMulAction F X :=
  inferInstanceAs (MulDistribMulAction F.toSubsemiring X)


/-- The action by a subfield is the action by the underlying field. -/
instance [Zero X] [SMulWithZero K X] (F : Subfield K) : SMulWithZero F X :=
  inferInstanceAs (SMulWithZero F.toSubsemiring X)


/-- The action by a subfield is the action by the underlying field. -/
instance [Zero X] [MulActionWithZero K X] (F : Subfield K) : MulActionWithZero F X :=
  inferInstanceAs (MulActionWithZero F.toSubsemiring X)


/-- The action by a subfield is the action by the underlying field. -/
instance [AddCommMonoid X] [Module K X] (F : Subfield K) : Module F X :=
  inferInstanceAs (Module F.toSubsemiring X)


/-- The action by a subfield is the action by the underlying field. -/
instance [Semiring X] [MulSemiringAction K X] (F : Subfield K) : MulSemiringAction F X :=
  inferInstanceAs (MulSemiringAction F.toSubsemiring X)


