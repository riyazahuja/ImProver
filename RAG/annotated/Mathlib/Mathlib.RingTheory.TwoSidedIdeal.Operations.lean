/--
The smallest two-sided ideal containing a set.
-/
abbrev span (s : Set R) : TwoSidedIdeal R :=
  { ringCon := ringConGen (fun a b ↦ a - b ∈ s) }


lemma subset_span {s : Set R} : s ⊆ (span s : Set R) := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    s : Set R
    ⊢ HasSubset.Subset s ↑(TwoSidedIdeal.span s)
  -/
  intro x hx
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    s : Set R
    x : R
    hx : Membership.mem s x
    ⊢ Membership.mem (↑(TwoSidedIdeal.span s)) x
  -/
  rw [SetLike.mem_coe, mem_iff]
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    s : Set R
    x : R
    hx : Membership.mem s x
    ⊢ (TwoSidedIdeal.span s).ringCon x 0
  -/
  exact RingConGen.Rel.of _ _ (by simpa using hx)
  /-
    🎉 no goals
  -/


lemma mem_span_iff {s : Set R} {x} :
    x ∈ span s ↔ ∀ (I : TwoSidedIdeal R), s ⊆ I → x ∈ I := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    s : Set R
    x : R
    ⊢ Iff (Membership.mem (TwoSidedIdeal.span s) x) (∀ (I : TwoSidedIdeal R), HasS …
  -/
  refine ⟨?_, fun h => h _ subset_span⟩
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    s : Set R
    x : R
    ⊢ Membership.mem (TwoSidedIdeal.span s) x → ∀ (I : TwoSidedIdeal R), HasSubset …
  -/
  delta span
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    s : Set R
    x : R
    ⊢ Membership.mem { ringCon := ringConGen fun a b => Membership.mem s (HSub.hSu …
  -/
  rw [RingCon.ringConGen_eq]
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    s : Set R
    x : R
    ⊢ Membership.mem { ringCon := InfSet.sInf (setOf fun s_1 => ∀ (x y : R), Membe …
  -/
  intro h I hI
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    s : Set R
    x : R
    h : Membership.mem { ringCon := InfSet.sInf (setOf fun s_1 => ∀ (x y : R), Mem …
    I : TwoSidedIdeal R
    hI : HasSubset.Subset s ↑I
    ⊢ Membership.mem I x
  -/
  refine sInf_le (α := RingCon R) ?_ h
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    s : Set R
    x : R
    h : Membership.mem { ringCon := InfSet.sInf (setOf fun s_1 => ∀ (x y : R), Mem …
    I : TwoSidedIdeal R
    hI : HasSubset.Subset s ↑I
    ⊢ Membership.mem (setOf fun s_1 => ∀ (x y : R), Membership.mem s (HSub.hSub x  …
  -/
  intro x y hxy
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    s : Set R
    x✝ : R
    h : Membership.mem { ringCon := InfSet.sInf (setOf fun s_1 => ∀ (x y : R), Mem …
    I : TwoSidedIdeal R
    hI : HasSubset.Subset s ↑I
    x y : R
    hxy : Membership.mem s (HSub.hSub x y)
    ⊢ I.ringCon x y
  -/
  specialize hI hxy
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    s : Set R
    x✝ : R
    h : Membership.mem { ringCon := InfSet.sInf (setOf fun s_1 => ∀ (x y : R), Mem …
    I : TwoSidedIdeal R
    x y : R
    hxy : Membership.mem s (HSub.hSub x y)
    hI : Membership.mem (↑I) (HSub.hSub x y)
    ⊢ I.ringCon x y
  -/
  rwa [SetLike.mem_coe, ← rel_iff] at hI
  /-
    🎉 no goals
  -/


lemma span_mono {s t : Set R} (h : s ⊆ t) : span s ≤ span t := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    s t : Set R
    h : HasSubset.Subset s t
    ⊢ LE.le (TwoSidedIdeal.span s) (TwoSidedIdeal.span t)
  -/
  intro x hx
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    s t : Set R
    h : HasSubset.Subset s t
    x : R
    hx : Membership.mem (TwoSidedIdeal.span s) x
    ⊢ Membership.mem (TwoSidedIdeal.span t) x
  -/
  rw [mem_span_iff] at hx ⊢
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    s t : Set R
    h : HasSubset.Subset s t
    x : R
    hx : ∀ (I : TwoSidedIdeal R), HasSubset.Subset s ↑I → Membership.mem I x
    ⊢ ∀ (I : TwoSidedIdeal R), HasSubset.Subset t ↑I → Membership.mem I x
  -/
  exact fun I hI => hx I <| h.trans hI
  /-
    🎉 no goals
  -/


/--
Pushout of a two-sided ideal. Defined as the span of the image of a two-sided ideal under a ring
homomorphism.
-/
def map (I : TwoSidedIdeal R) : TwoSidedIdeal S :=
  span (f '' I)


lemma map_mono {I J : TwoSidedIdeal R} (h : I ≤ J) :
    map f I ≤ map f J :=
  span_mono <| Set.image_mono h


/--
Preimage of a two-sided ideal, as a two-sided ideal. -/
def comap (I : TwoSidedIdeal S) : TwoSidedIdeal R where
  ringCon := I.ringCon.comap f


lemma mem_comap {I : TwoSidedIdeal S} {x : R} :
    x ∈ I.comap f ↔ f x ∈ I := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : NonUnitalNonAssocRing R
    inst✝² : NonUnitalNonAssocRing S
    F : Type u_3
    inst✝¹ : FunLike F R S
    f : F
    inst✝ : NonUnitalRingHomClass F R S
    I : TwoSidedIdeal S
    x : R
    ⊢ Iff (Membership.mem (TwoSidedIdeal.comap f I) x) (Membership.mem I (f x))
  -/
  simp [comap, RingCon.comap, mem_iff]
  /-
    🎉 no goals
  -/



open AddSubgroup in
/-- If `s : Set R` is absorbing under multiplication, then its `TwoSidedIdeal.span` coincides with
its `AddSubgroup.closure`, as sets. -/
lemma mem_span_iff_mem_addSubgroup_closure_absorbing {s : Set R}
    (h_left : ∀ x y, y ∈ s → x * y ∈ s) (h_right : ∀ y x, y ∈ s → y * x ∈ s) {z : R} :
    z ∈ span s ↔ z ∈ closure s := by
  have h_left' {x y} (hy : y ∈ closure s) : x * y ∈ closure s := by
    have := (AddMonoidHom.mulLeft x).map_closure s ▸ mem_map_of_mem _ hy
    refine closure_mono ?_ this
    rintro - ⟨y, hy, rfl⟩
    exact h_left x y hy
  have h_right' {y x} (hy : y ∈ closure s) : y * x ∈ closure s := by
    have := (AddMonoidHom.mulRight x).map_closure s ▸ mem_map_of_mem _ hy
    refine closure_mono ?_ this
    rintro - ⟨y, hy, rfl⟩
    exact h_right y x hy
  let I : TwoSidedIdeal R := .mk' (closure s) (AddSubgroup.zero_mem _)
    (AddSubgroup.add_mem _) (AddSubgroup.neg_mem _) h_left' h_right'
  /-
    R : Type u_1
    inst✝ : NonUnitalRing R
    s : Set R
    h_left : ∀ (x y : R), Membership.mem s y → Membership.mem s (HMul.hMul x y)
    h_right : ∀ (y x : R), Membership.mem s y → Membership.mem s (HMul.hMul y x)
    z : R
    h_left' : ∀ {x y : R}, Membership.mem (AddSubgroup.closure s) y → Membership.m …
    h_right' : ∀ {y x : R}, Membership.mem (AddSubgroup.closure s) y → Membership. …
    I : TwoSidedIdeal R := TwoSidedIdeal.mk' ↑(AddSubgroup.closure s) ⋯ ⋯ ⋯ ⋯ ⋯
    ⊢ Iff (Membership.mem (TwoSidedIdeal.span s) z) (Membership.mem (AddSubgroup.c …
  -/
  suffices z ∈ span s ↔ z ∈ I by simpa only [I, mem_mk', SetLike.mem_coe]
  /-
    R : Type u_1
    inst✝ : NonUnitalRing R
    s : Set R
    h_left : ∀ (x y : R), Membership.mem s y → Membership.mem s (HMul.hMul x y)
    h_right : ∀ (y x : R), Membership.mem s y → Membership.mem s (HMul.hMul y x)
    z : R
    h_left' : ∀ {x y : R}, Membership.mem (AddSubgroup.closure s) y → Membership.m …
    h_right' : ∀ {y x : R}, Membership.mem (AddSubgroup.closure s) y → Membership. …
    I : TwoSidedIdeal R := TwoSidedIdeal.mk' ↑(AddSubgroup.closure s) ⋯ ⋯ ⋯ ⋯ ⋯
    ⊢ Iff (Membership.mem (TwoSidedIdeal.span s) z) (Membership.mem I z)
  -/
  rw [mem_span_iff]
  -- Suppose that for every ideal `J` with `s ⊆ J`, then `z ∈ J`. Apply this to `I` to get `z ∈ I`.
  /-
    R : Type u_1
    inst✝ : NonUnitalRing R
    s : Set R
    h_left : ∀ (x y : R), Membership.mem s y → Membership.mem s (HMul.hMul x y)
    h_right : ∀ (y x : R), Membership.mem s y → Membership.mem s (HMul.hMul y x)
    z : R
    h_left' : ∀ {x y : R}, Membership.mem (AddSubgroup.closure s) y → Membership.m …
    h_right' : ∀ {y x : R}, Membership.mem (AddSubgroup.closure s) y → Membership. …
    I : TwoSidedIdeal R := TwoSidedIdeal.mk' ↑(AddSubgroup.closure s) ⋯ ⋯ ⋯ ⋯ ⋯
    ⊢ Iff (∀ (I : TwoSidedIdeal R), HasSubset.Subset s ↑I → Membership.mem I z) (M …
  -/
  refine ⟨fun h ↦ h I fun x hx ↦ ?mem_closure_of_forall, fun hz J hJ ↦ ?mem_ideal_of_subset⟩
  /-
    case mem_closure_of_forall
    R : Type u_1
    inst✝ : NonUnitalRing R
    s : Set R
    h_left : ∀ (x y : R), Membership.mem s y → Membership.mem s (HMul.hMul x y)
    h_right : ∀ (y x : R), Membership.mem s y → Membership.mem s (HMul.hMul y x)
    z : R
    h_left' : ∀ {x y : R}, Membership.mem (AddSubgroup.closure s) y → Membership.m …
    h_right' : ∀ {y x : R}, Membership.mem (AddSubgroup.closure s) y → Membership. …
    I : TwoSidedIdeal R := TwoSidedIdeal.mk' ↑(AddSubgroup.closure s) ⋯ ⋯ ⋯ ⋯ ⋯
    h : ∀ (I : TwoSidedIdeal R), HasSubset.Subset s ↑I → Membership.mem I z
    x : R
    hx : Membership.mem s x
    ⊢ Membership.mem (↑I) x
  -/
  case mem_closure_of_forall => simpa only [I, SetLike.mem_coe, mem_mk'] using subset_closure hx
  /- Conversely, suppose that `z ∈ I` and that `J` is any ideal containing `s`. Then by the
  induction principle for `AddSubgroup`, we must also have `z ∈ J`. -/
  case mem_ideal_of_subset =>
    simp only [I, SetLike.mem_coe, mem_mk'] at hz
    induction hz using closure_induction with
    | mem x hx => exact hJ hx
    | one => exact zero_mem _
    | mul x y _ _ hx hy => exact J.add_mem hx hy
    | inv x _ hx => exact J.neg_mem hx


lemma set_mul_subset {s : Set R} {I : TwoSidedIdeal R} (h : s ⊆ I) (t : Set R):
    t * s ⊆ I := by
  /-
    R : Type u_1
    inst✝ : NonUnitalRing R
    s : Set R
    I : TwoSidedIdeal R
    h : HasSubset.Subset s ↑I
    t : Set R
    ⊢ HasSubset.Subset (HMul.hMul t s) ↑I
  -/
  rintro - ⟨r, -, x, hx, rfl⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝ : NonUnitalRing R
    s : Set R
    I : TwoSidedIdeal R
    h : HasSubset.Subset s ↑I
    t : Set R
    r x : R
    hx : Membership.mem s x
    ⊢ Membership.mem (↑I) ((fun x1 x2 => HMul.hMul x1 x2) r x)
  -/
  exact mul_mem_left _ _ _ (h hx)
  /-
    🎉 no goals
  -/


lemma subset_mul_set {s : Set R} {I : TwoSidedIdeal R} (h : s ⊆ I) (t : Set R):
    s * t ⊆ I := by
  /-
    R : Type u_1
    inst✝ : NonUnitalRing R
    s : Set R
    I : TwoSidedIdeal R
    h : HasSubset.Subset s ↑I
    t : Set R
    ⊢ HasSubset.Subset (HMul.hMul s t) ↑I
  -/
  rintro - ⟨x, hx, r, -, rfl⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝ : NonUnitalRing R
    s : Set R
    I : TwoSidedIdeal R
    h : HasSubset.Subset s ↑I
    t : Set R
    x : R
    hx : Membership.mem s x
    r : R
    ⊢ Membership.mem (↑I) ((fun x1 x2 => HMul.hMul x1 x2) x r)
  -/
  exact mul_mem_right _ _ _ (h hx)
  /-
    🎉 no goals
  -/


lemma mem_span_iff_mem_addSubgroup_closure_nonunital {s : Set R} {z : R} :
    z ∈ span s ↔ z ∈ AddSubgroup.closure (s ∪ s * univ ∪ univ * s ∪ univ * s * univ) := by
  /-
    R : Type u_1
    inst✝ : NonUnitalRing R
    s : Set R
    z : R
    ⊢ Iff (Membership.mem (TwoSidedIdeal.span s) z) (Membership.mem (AddSubgroup.c …
  -/
  trans z ∈ span (s ∪ s * univ ∪ univ * s ∪ univ * s * univ)
    /-
      R : Type u_1
      inst✝ : NonUnitalRing R
      s : Set R
      z : R
      ⊢ Iff (Membership.mem (TwoSidedIdeal.span s) z) (Membership.mem (TwoSidedIdeal …
    -/
  · refine ⟨(span_mono (by simp only [Set.union_assoc, Set.subset_union_left]) ·), fun h ↦ ?_⟩
    /-
      R : Type u_1
      inst✝ : NonUnitalRing R
      s : Set R
      z : R
      h : Membership.mem (TwoSidedIdeal.span (Union.union (Union.union (Union.union  …
      ⊢ Membership.mem (TwoSidedIdeal.span s) z
    -/
    refine mem_span_iff.mp h (span s) ?_
    /-
      R : Type u_1
      inst✝ : NonUnitalRing R
      s : Set R
      z : R
      h : Membership.mem (TwoSidedIdeal.span (Union.union (Union.union (Union.union  …
      ⊢ HasSubset.Subset (Union.union (Union.union (Union.union s (HMul.hMul s Set.u …
    -/
    simp only [union_subset_iff, union_assoc]
    exact ⟨subset_span, subset_mul_set subset_span _, set_mul_subset subset_span _,
      subset_mul_set (set_mul_subset subset_span _) _⟩
    /-
      R : Type u_1
      inst✝ : NonUnitalRing R
      s : Set R
      z : R
      ⊢ Iff (Membership.mem (TwoSidedIdeal.span (Union.union (Union.union (Union.uni …
    -/
  · refine mem_span_iff_mem_addSubgroup_closure_absorbing ?_ ?_
    · rintro x y (((hy | ⟨y, hy, r, -, rfl⟩) | ⟨r, -, y, hy, rfl⟩) |
        ⟨-, ⟨r', -, y, hy, rfl⟩, r, -, rfl⟩)
        /-
          case refine_1.inl.inl.inl
          R : Type u_1
          inst✝ : NonUnitalRing R
          s : Set R
          z x y : R
          hy : Membership.mem s y
          ⊢ Membership.mem (Union.union (Union.union (Union.union s (HMul.hMul s Set.uni …
        -/
      · exact .inl <| .inr <| ⟨x, mem_univ _, y, hy, rfl⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_1.inl.inl.inr.intro.intro.intro.intro
          R : Type u_1
          inst✝ : NonUnitalRing R
          s : Set R
          z x y : R
          hy : Membership.mem s y
          r : R
          ⊢ Membership.mem (Union.union (Union.union (Union.union s (HMul.hMul s Set.uni …
        -/
      · exact .inr <| ⟨x * y, ⟨x, mem_univ _, y, hy, rfl⟩, r, mem_univ _, mul_assoc ..⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_1.inl.inr.intro.intro.intro.intro
          R : Type u_1
          inst✝ : NonUnitalRing R
          s : Set R
          z x r y : R
          hy : Membership.mem s y
          ⊢ Membership.mem (Union.union (Union.union (Union.union s (HMul.hMul s Set.uni …
        -/
      · exact .inl <| .inr <| ⟨x * r, mem_univ _, y, hy, mul_assoc ..⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_1.inr.intro.intro.intro.intro.intro.intro.intro.intro
          R : Type u_1
          inst✝ : NonUnitalRing R
          s : Set R
          z x r' y : R
          hy : Membership.mem s y
          r : R
          ⊢ Membership.mem (Union.union (Union.union (Union.union s (HMul.hMul s Set.uni …
        -/
      · refine .inr <| ⟨x * r' * y, ⟨x * r', mem_univ _, y, hy, ?_⟩, ⟨r, mem_univ _, ?_⟩⟩
        /-
          case refine_1.inr.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
          R : Type u_1
          inst✝ : NonUnitalRing R
          s : Set R
          z x r' y : R
          hy : Membership.mem s y
          r : R
          ⊢ Eq ((fun x1 x2 => HMul.hMul x1 x2) (HMul.hMul x r') y) (HMul.hMul (HMul.hMul …
        -/
        all_goals simp [mul_assoc]
        /-
          🎉 no goals
        -/
    · rintro y x (((hy | ⟨y, hy, r, -, rfl⟩) | ⟨r, -, y, hy, rfl⟩) |
        ⟨-, ⟨r', -, y, hy, rfl⟩, r, -, rfl⟩)
        /-
          case refine_2.inl.inl.inl
          R : Type u_1
          inst✝ : NonUnitalRing R
          s : Set R
          z y x : R
          hy : Membership.mem s y
          ⊢ Membership.mem (Union.union (Union.union (Union.union s (HMul.hMul s Set.uni …
        -/
      · exact .inl <| .inl <| .inr ⟨y, hy, x, mem_univ _, rfl⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_2.inl.inl.inr.intro.intro.intro.intro
          R : Type u_1
          inst✝ : NonUnitalRing R
          s : Set R
          z x y : R
          hy : Membership.mem s y
          r : R
          ⊢ Membership.mem (Union.union (Union.union (Union.union s (HMul.hMul s Set.uni …
        -/
      · exact .inl <| .inl <| .inr ⟨y, hy, r * x, mem_univ _, (mul_assoc ..).symm⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_2.inl.inr.intro.intro.intro.intro
          R : Type u_1
          inst✝ : NonUnitalRing R
          s : Set R
          z x r y : R
          hy : Membership.mem s y
          ⊢ Membership.mem (Union.union (Union.union (Union.union s (HMul.hMul s Set.uni …
        -/
      · exact .inr <| ⟨r * y, ⟨r, mem_univ _, y, hy, rfl⟩, x, mem_univ _, rfl⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_2.inr.intro.intro.intro.intro.intro.intro.intro.intro
          R : Type u_1
          inst✝ : NonUnitalRing R
          s : Set R
          z x r' y : R
          hy : Membership.mem s y
          r : R
          ⊢ Membership.mem (Union.union (Union.union (Union.union s (HMul.hMul s Set.uni …
        -/
      · refine .inr <| ⟨r' * y, ⟨r', mem_univ _, y, hy, rfl⟩, r * x, mem_univ _, ?_⟩
        /-
          case refine_2.inr.intro.intro.intro.intro.intro.intro.intro.intro
          R : Type u_1
          inst✝ : NonUnitalRing R
          s : Set R
          z x r' y : R
          hy : Membership.mem s y
          r : R
          ⊢ Eq ((fun x1 x2 => HMul.hMul x1 x2) (HMul.hMul r' y) (HMul.hMul r x)) (HMul.h …
        -/
        simp [mul_assoc]
        /-
          🎉 no goals
        -/


open Pointwise Set in
lemma mem_span_iff_mem_addSubgroup_closure {s : Set R} {z : R} :
    z ∈ span s ↔ z ∈ AddSubgroup.closure (univ * s * univ) := by
  /-
    R : Type u_1
    inst✝ : Ring R
    s : Set R
    z : R
    ⊢ Iff (Membership.mem (TwoSidedIdeal.span s) z) (Membership.mem (AddSubgroup.c …
  -/
  trans z ∈ span (univ * s * univ)
    /-
      R : Type u_1
      inst✝ : Ring R
      s : Set R
      z : R
      ⊢ Iff (Membership.mem (TwoSidedIdeal.span s) z) (Membership.mem (TwoSidedIdeal …
    -/
  · refine ⟨(span_mono (fun x hx ↦ ?_) ·), fun hz ↦ ?_⟩
      /-
        case refine_1
        R : Type u_1
        inst✝ : Ring R
        s : Set R
        z : R
        x✝ : Membership.mem (TwoSidedIdeal.span s) z
        x : R
        hx : Membership.mem s x
        ⊢ Membership.mem (HMul.hMul (HMul.hMul Set.univ s) Set.univ) x
      -/
    · exact ⟨1 * x, ⟨1, mem_univ _, x, hx, rfl⟩, 1, mem_univ _, by simp⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        R : Type u_1
        inst✝ : Ring R
        s : Set R
        z : R
        hz : Membership.mem (TwoSidedIdeal.span (HMul.hMul (HMul.hMul Set.univ s) Set. …
        ⊢ Membership.mem (TwoSidedIdeal.span s) z
      -/
    · exact mem_span_iff.mp hz (span s) <| subset_mul_set (set_mul_subset subset_span _) _
      /-
        🎉 no goals
      -/
    /-
      R : Type u_1
      inst✝ : Ring R
      s : Set R
      z : R
      ⊢ Iff (Membership.mem (TwoSidedIdeal.span (HMul.hMul (HMul.hMul Set.univ s) Se …
    -/
  · refine mem_span_iff_mem_addSubgroup_closure_absorbing ?_ ?_
      /-
        case refine_1
        R : Type u_1
        inst✝ : Ring R
        s : Set R
        z : R
        ⊢ ∀ (x y : R), Membership.mem (HMul.hMul (HMul.hMul Set.univ s) Set.univ) y →  …
      -/
    · intro x y hy
      /-
        case refine_1
        R : Type u_1
        inst✝ : Ring R
        s : Set R
        z x y : R
        hy : Membership.mem (HMul.hMul (HMul.hMul Set.univ s) Set.univ) y
        ⊢ Membership.mem (HMul.hMul (HMul.hMul Set.univ s) Set.univ) (HMul.hMul x y)
      -/
      rw [mul_assoc] at hy ⊢
      /-
        case refine_1
        R : Type u_1
        inst✝ : Ring R
        s : Set R
        z x y : R
        hy : Membership.mem (HMul.hMul Set.univ (HMul.hMul s Set.univ)) y
        ⊢ Membership.mem (HMul.hMul Set.univ (HMul.hMul s Set.univ)) (HMul.hMul x y)
      -/
      obtain ⟨r, -, y, hy, rfl⟩ := hy
      /-
        case refine_1.intro.intro.intro.intro
        R : Type u_1
        inst✝ : Ring R
        s : Set R
        z x r y : R
        hy : Membership.mem (HMul.hMul s Set.univ) y
        ⊢ Membership.mem (HMul.hMul Set.univ (HMul.hMul s Set.univ)) (HMul.hMul x ((fu …
      -/
      exact ⟨x * r, mem_univ _, y, hy, mul_assoc ..⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        R : Type u_1
        inst✝ : Ring R
        s : Set R
        z : R
        ⊢ ∀ (y x : R), Membership.mem (HMul.hMul (HMul.hMul Set.univ s) Set.univ) y →  …
      -/
    · rintro - x ⟨y, hy, r, -, rfl⟩
      /-
        case refine_2.intro.intro.intro.intro
        R : Type u_1
        inst✝ : Ring R
        s : Set R
        z x y : R
        hy : Membership.mem (HMul.hMul Set.univ s) y
        r : R
        ⊢ Membership.mem (HMul.hMul (HMul.hMul Set.univ s) Set.univ) (HMul.hMul ((fun  …
      -/
      exact ⟨y, hy, r * x, mem_univ _, (mul_assoc ..).symm⟩
      /-
        🎉 no goals
      -/


instance : SMul R I where smul r x := ⟨r • x.1, I.mul_mem_left _ _ x.2⟩


instance : SMul Rᵐᵒᵖ I where smul r x := ⟨r • x.1, I.mul_mem_right _ _ x.2⟩


instance leftModule : Module R I :=
  Function.Injective.module _ (coeAddMonoidHom I) Subtype.coe_injective fun _ _ ↦ rfl


@[simp]
lemma coe_smul {r : R} {x : I} : (r • x : R) = r * (x : R) := rfl


instance rightModule : Module Rᵐᵒᵖ I :=
  Function.Injective.module _ (coeAddMonoidHom I) Subtype.coe_injective fun _ _ ↦ rfl


@[simp]
lemma coe_mop_smul {r : Rᵐᵒᵖ} {x : I} : (r • x : R) = (x : R) * r.unop := rfl


instance : SMulCommClass R Rᵐᵒᵖ I where
  smul_comm r s x := Subtype.ext <| smul_comm r s x.1


/--
For any `I : RingCon R`, when we view it as an ideal, `I.subtype` is the injective `R`-linear map
`I → R`.
-/
@[simps]
def subtype : I →ₗ[R] R where
  toFun x := x.1
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


/--
For any `RingCon R`, when we view it as an ideal in `Rᵒᵖ`, `subtype` is the injective `Rᵐᵒᵖ`-linear
map `I → Rᵐᵒᵖ`.
-/
@[simps]
def subtypeMop : I →ₗ[Rᵐᵒᵖ] Rᵐᵒᵖ where
  toFun x := MulOpposite.op x.1
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


/-- Given an ideal `I`, `span I` is the smallest two-sided ideal containing `I`. -/
def fromIdeal : Ideal R →o TwoSidedIdeal R where
  toFun I := span I
  monotone' _ _ := span_mono


lemma mem_fromIdeal {I : Ideal R} {x : R} :
                                       /-
                                         R : Type u_1
                                         inst✝ : Ring R
                                         I : Ideal R
                                         x : R
                                         ⊢ Iff (Membership.mem (TwoSidedIdeal.fromIdeal I) x) (Membership.mem (TwoSided …
                                       -/
    x ∈ fromIdeal I ↔ x ∈ span I := by simp [fromIdeal]
                                       /-
                                         🎉 no goals
                                       -/


/-- Every two-sided ideal is also a left ideal. -/
def asIdeal : TwoSidedIdeal R →o Ideal R where
  toFun I :=
  { carrier := I
    add_mem' := I.add_mem
    zero_mem' := I.zero_mem
    smul_mem' := fun r x hx => I.mul_mem_left r x hx }
  monotone' _ _ h _ h' := h h'


@[simp]
lemma mem_asIdeal {I : TwoSidedIdeal R} {x : R} :
                                /-
                                  R : Type u_1
                                  inst✝ : Ring R
                                  I : TwoSidedIdeal R
                                  x : R
                                  ⊢ Iff (Membership.mem (TwoSidedIdeal.asIdeal I) x) (Membership.mem I x)
                                -/
    x ∈ asIdeal I ↔ x ∈ I := by simp [asIdeal]
                                /-
                                  🎉 no goals
                                -/


lemma gc : GaloisConnection fromIdeal (asIdeal (R := R)) :=
  fun I J => ⟨fun h x hx ↦ h <| mem_span_iff.2 fun _ H ↦ H hx, fun h x hx ↦ by
    /-
      R : Type u_1
      inst✝ : Ring R
      I : Ideal R
      J : TwoSidedIdeal R
      h : LE.le I (TwoSidedIdeal.asIdeal J)
      x : R
      hx : Membership.mem (TwoSidedIdeal.fromIdeal I) x
      ⊢ Membership.mem J x
    -/
    simp only [fromIdeal, OrderHom.coe_mk, mem_span_iff] at hx
    /-
      R : Type u_1
      inst✝ : Ring R
      I : Ideal R
      J : TwoSidedIdeal R
      h : LE.le I (TwoSidedIdeal.asIdeal J)
      x : R
      hx : ∀ (I_1 : TwoSidedIdeal R), HasSubset.Subset ↑I ↑I_1 → Membership.mem I_1 x
      ⊢ Membership.mem J x
    -/
    exact hx _ h⟩
    /-
      🎉 no goals
    -/


@[simp]
lemma coe_asIdeal {I : TwoSidedIdeal R} : (asIdeal I : Set R) = I := rfl


/-- Every two-sided ideal is also a right ideal. -/
def asIdealOpposite : TwoSidedIdeal R →o Ideal Rᵐᵒᵖ where
  toFun I := asIdeal ⟨I.ringCon.op⟩
  monotone' I J h x h' := by
    /-
      R : Type u_1
      inst✝ : Ring R
      I✝ I J : TwoSidedIdeal R
      h : LE.le I J
      x : MulOpposite R
      h' : Membership.mem ((fun I => TwoSidedIdeal.asIdeal { ringCon := I.ringCon.op …
      ⊢ Membership.mem ((fun I => TwoSidedIdeal.asIdeal { ringCon := I.ringCon.op }) …
    -/
    simp only [mem_asIdeal, mem_iff, RingCon.op_iff, MulOpposite.unop_zero] at h' ⊢
    /-
      R : Type u_1
      inst✝ : Ring R
      I✝ I J : TwoSidedIdeal R
      h : LE.le I J
      x : MulOpposite R
      h' : I.ringCon 0 (MulOpposite.unop x)
      ⊢ J.ringCon 0 (MulOpposite.unop x)
    -/
    exact J.rel_iff _ _ |>.2 <| h <| I.rel_iff 0 x.unop |>.1 h'
    /-
      🎉 no goals
    -/


lemma mem_asIdealOpposite {I : TwoSidedIdeal R} {x : Rᵐᵒᵖ} :
    x ∈ asIdealOpposite I ↔ x.unop ∈ I := by
  simpa [asIdealOpposite, asIdeal, TwoSidedIdeal.mem_iff, RingCon.op_iff] using
    ⟨I.ringCon.symm, I.ringCon.symm⟩


/--
When the ring is commutative, two-sided ideals are exactly the same as left ideals.
-/
def orderIsoIdeal : TwoSidedIdeal R ≃o Ideal R where
  toFun := asIdeal
  invFun := fromIdeal
  map_rel_iff' := ⟨fun h _ hx ↦ h hx, fun h ↦ asIdeal.monotone' h⟩
                                                             /-
                                                               R : Type u_1
                                                               inst✝ : CommRing R
                                                               x✝¹ : TwoSidedIdeal R
                                                               x✝ : R
                                                               ⊢ Iff (∀ (I : TwoSidedIdeal R), HasSubset.Subset ↑(TwoSidedIdeal.asIdeal x✝¹)  …
                                                             -/
  left_inv _ := SetLike.ext fun _ ↦ mem_span_iff.trans <| by aesop
                                                             /-
                                                               🎉 no goals
                                                             -/
  right_inv J := SetLike.ext fun x ↦ mem_span_iff.trans
    ⟨fun h ↦ mem_mk' _ _ _ _ _ _ _ |>.1 <| h (mk'
      J J.zero_mem J.add_mem J.neg_mem (J.mul_mem_left _) (J.mul_mem_right _))
                   /-
                     R : Type u_1
                     inst✝ : CommRing R
                     J : Ideal R
                     x✝ : R
                     h : ∀ (I : TwoSidedIdeal R), HasSubset.Subset ↑J ↑I → Membership.mem I x✝
                     x : R
                     ⊢ Membership.mem (↑J) x → Membership.mem (↑(TwoSidedIdeal.mk' ↑J ⋯ ⋯ ⋯ ⋯ ⋯)) x
                   -/
                   /-
                     🎉 no goals
                   -/
      (fun x => by simp), by aesop⟩
                             /-
                               🎉 no goals
                             -/


/-- Bundle an `Ideal` that is already two-sided as a `TwoSidedIdeal`. -/
def toTwoSided (I : Ideal R) (mul_mem_right : ∀ {x y}, x ∈ I → x * y ∈ I) : TwoSidedIdeal R :=
  TwoSidedIdeal.mk' I I.zero_mem I.add_mem I.neg_mem (I.smul_mem _) mul_mem_right


@[simp]
lemma mem_toTwoSided {I : Ideal R} {h} {x : R} :
    x ∈ I.toTwoSided h ↔ x ∈ I := by
  /-
    R : Type u_1
    inst✝ : Ring R
    I : Ideal R
    h : ∀ {x y : R}, Membership.mem I x → Membership.mem I (HMul.hMul x y)
    x : R
    ⊢ Iff (Membership.mem (I.toTwoSided h) x) (Membership.mem I x)
  -/
  simp [toTwoSided]
  /-
    🎉 no goals
  -/


@[simp]
lemma coe_toTwoSided (I : Ideal R) (h) : (I.toTwoSided h : Set R) = I := by
  /-
    R : Type u_1
    inst✝ : Ring R
    I : Ideal R
    h : ∀ {x y : R}, Membership.mem I x → Membership.mem I (HMul.hMul x y)
    ⊢ Eq ↑(I.toTwoSided h) ↑I
  -/
  simp [toTwoSided]
  /-
    🎉 no goals
  -/


@[simp]
lemma toTwoSided_asIdeal (I : TwoSidedIdeal R) (h) : (TwoSidedIdeal.asIdeal I).toTwoSided h = I :=
     /-
       R : Type u_1
       inst✝ : Ring R
       I : TwoSidedIdeal R
       h : ∀ {x y : R}, Membership.mem (TwoSidedIdeal.asIdeal I) x → Membership.mem ( …
       ⊢ Eq ((TwoSidedIdeal.asIdeal I).toTwoSided h) I
     -/
  by ext; simp
          /-
            🎉 no goals
          -/


@[simp]
lemma asIdeal_toTwoSided (I : Ideal R) (h) : TwoSidedIdeal.asIdeal (I.toTwoSided h) = I := by
  /-
    R : Type u_1
    inst✝ : Ring R
    I : Ideal R
    h : ∀ {x y : R}, Membership.mem I x → Membership.mem I (HMul.hMul x y)
    ⊢ Eq (TwoSidedIdeal.asIdeal (I.toTwoSided h)) I
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝ : Ring R
    I : Ideal R
    h : ∀ {x y : R}, Membership.mem I x → Membership.mem I (HMul.hMul x y)
    x✝ : R
    ⊢ Iff (Membership.mem (TwoSidedIdeal.asIdeal (I.toTwoSided h)) x✝) (Membership …
  -/
  simp
  /-
    🎉 no goals
  -/


instance : CanLift (Ideal R) (TwoSidedIdeal R) TwoSidedIdeal.asIdeal
    (fun I => ∀ {x y}, x ∈ I → x * y ∈ I) where
  prf I mul_mem_right := ⟨I.toTwoSided mul_mem_right, asIdeal_toTwoSided ..⟩


