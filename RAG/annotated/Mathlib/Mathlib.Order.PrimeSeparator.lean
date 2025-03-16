lemma mem_ideal_sup_principal (a b : α) (J : Ideal α) : b ∈ J ⊔ principal a ↔ ∃ j ∈ J, b ≤ j ⊔ a :=
  ⟨fun ⟨j, ⟨jJ, _, ha', bja'⟩⟩ => ⟨j, jJ, le_trans bja' (sup_le_sup_left ha' j)⟩,
    fun ⟨j, hj, hbja⟩ => ⟨j, hj, a, le_refl a, hbja⟩⟩


theorem prime_ideal_of_disjoint_filter_ideal (hFI : Disjoint (F : Set α) (I : Set α)) :
    ∃ J : Ideal α, (IsPrime J) ∧ I ≤ J ∧ Disjoint (F : Set α) J := by

  -- Let S be the set of ideals containing I and disjoint from F.
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    ⊢ Exists fun J => And J.IsPrime (And (LE.le I J) (Disjoint ↑F ↑J))
  -/
  set S : Set (Set α) := { J : Set α | IsIdeal J ∧ I ≤ J ∧ Disjoint (F : Set α) J }

  -- Then I is in S...
  have IinS : ↑I ∈ S := by
    refine ⟨Order.Ideal.isIdeal I, by trivial⟩

  -- ...and S contains upper bounds for any non-empty chains.
  have chainub : ∀ c ⊆ S, IsChain (· ⊆ ·) c → c.Nonempty → ∃ ub ∈ S, ∀ s ∈ c, s ⊆ ub := by
    intros c hcS hcC hcNe
    use sUnion c
    refine ⟨?_, fun s hs ↦ le_sSup hs⟩
    simp only [le_eq_subset, mem_setOf_eq, disjoint_sUnion_right, S]
    let ⟨J, hJ⟩ := hcNe
    refine ⟨Order.isIdeal_sUnion_of_isChain (fun _ hJ ↦ (hcS hJ).1) hcC hcNe,
            ⟨le_trans (hcS hJ).2.1 (le_sSup hJ), fun J hJ ↦ (hcS hJ).2.2⟩⟩

  -- Thus, by Zorn's lemma, we can pick a maximal ideal J in S.
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    IinS : Membership.mem S ↑I
    chainub : ∀ (c : Set (Set α)), HasSubset.Subset c S → IsChain (fun x1 x2 => Ha …
    ⊢ Exists fun J => And J.IsPrime (And (LE.le I J) (Disjoint ↑F ↑J))
  -/
  obtain ⟨Jset, _, hmax⟩ := zorn_subset_nonempty S chainub I IinS
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    IinS : Membership.mem S ↑I
    chainub : ∀ (c : Set (Set α)), HasSubset.Subset c S → IsChain (fun x1 x2 => Ha …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    ⊢ Exists fun J => And J.IsPrime (And (LE.le I J) (Disjoint ↑F ↑J))
  -/
  obtain ⟨Jidl, IJ, JF⟩ := hmax.prop
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    IinS : Membership.mem S ↑I
    chainub : ∀ (c : Set (Set α)), HasSubset.Subset c S → IsChain (fun x1 x2 => Ha …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    ⊢ Exists fun J => And J.IsPrime (And (LE.le I J) (Disjoint ↑F ↑J))
  -/
  set J := IsIdeal.toIdeal Jidl
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    IinS : Membership.mem S ↑I
    chainub : ∀ (c : Set (Set α)), HasSubset.Subset c S → IsChain (fun x1 x2 => Ha …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    ⊢ Exists fun J => And J.IsPrime (And (LE.le I J) (Disjoint ↑F ↑J))
  -/
  use J
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    IinS : Membership.mem S ↑I
    chainub : ∀ (c : Set (Set α)), HasSubset.Subset c S → IsChain (fun x1 x2 => Ha …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    ⊢ And J.IsPrime (And (LE.le I J) (Disjoint ↑F ↑J))
  -/
  have IJ' : I ≤ J := IJ

  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    IinS : Membership.mem S ↑I
    chainub : ∀ (c : Set (Set α)), HasSubset.Subset c S → IsChain (fun x1 x2 => Ha …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    ⊢ And J.IsPrime (And (LE.le I J) (Disjoint ↑F ↑J))
  -/
  clear chainub IinS

  -- By construction, J contains I and is disjoint from F. It remains to prove that J is prime.
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    ⊢ And J.IsPrime (And (LE.le I J) (Disjoint ↑F ↑J))
  -/
  refine ⟨?_, ⟨IJ, JF⟩⟩

  -- First note that J is proper: ⊤ ∈ F so ⊤ ∉ J because F and J are disjoint.
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    ⊢ J.IsPrime
  -/
  have Jpr : IsProper J := isProper_of_not_mem (Set.disjoint_left.1 JF F.top_mem)

  -- Suppose that a₁ ∉ J, a₂ ∉ J. We need to prove that a₁ ⊔ a₂ ∉ J.
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    ⊢ J.IsPrime
  -/
  rw [isPrime_iff_mem_or_mem]
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    ⊢ ∀ {x y : α}, Membership.mem J (Min.min x y) → Or (Membership.mem J x) (Membe …
  -/
  intros a₁ a₂
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ⊢ Membership.mem J (Min.min a₁ a₂) → Or (Membership.mem J a₁) (Membership.mem  …
  -/
  contrapose!
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ⊢ And (Not (Membership.mem J a₁)) (Not (Membership.mem J a₂)) → Not (Membershi …
  -/
  intro ⟨ha₁, ha₂⟩

  -- Consider the ideals J₁, J₂ generated by J ∪ {a₁} and J ∪ {a₂}, respectively.
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  let J₁ := J ⊔ principal a₁
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  let J₂ := J ⊔ principal a₂

  -- For each i, Jᵢ is an ideal that contains aᵢ, and is not equal to J.

  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  have a₁J₁ : a₁ ∈ J₁ := mem_of_subset_of_mem (le_sup_right : _ ≤ J ⊔ _) mem_principal_self
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  have a₂J₂ : a₂ ∈ J₂ := mem_of_subset_of_mem (le_sup_right : _ ≤ J ⊔ _) mem_principal_self
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    a₂J₂ : Membership.mem J₂ a₂
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  have J₁J : ↑J₁ ≠ Jset := ne_of_mem_of_not_mem' a₁J₁ ha₁
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    a₂J₂ : Membership.mem J₂ a₂
    J₁J : Ne (↑J₁) Jset
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  have J₂J : ↑J₂ ≠ Jset := ne_of_mem_of_not_mem' a₂J₂ ha₂

  -- Therefore, since J is maximal, we must have Jᵢ ∉ S.
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    a₂J₂ : Membership.mem J₂ a₂
    J₁J : Ne (↑J₁) Jset
    J₂J : Ne (↑J₂) Jset
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  have J₁S : ↑J₁ ∉ S := fun h => J₁J (hmax.eq_of_le h (le_sup_left : J ≤ J₁)).symm
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    a₂J₂ : Membership.mem J₂ a₂
    J₁J : Ne (↑J₁) Jset
    J₂J : Ne (↑J₂) Jset
    J₁S : Not (Membership.mem S ↑J₁)
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  have J₂S : ↑J₂ ∉ S := fun h => J₂J (hmax.eq_of_le h (le_sup_left : J ≤ J₂)).symm

  -- Since Jᵢ is an ideal that contains I, we have that Jᵢ is not disjoint from F.
  have J₁F : ¬ (Disjoint (F : Set α) J₁) := by
    intro hdis
    apply J₁S
    simp only [le_eq_subset, mem_setOf_eq, SetLike.coe_subset_coe, S]
    exact ⟨J₁.isIdeal, le_trans IJ' le_sup_left, hdis⟩

  have J₂F : ¬ (Disjoint (F : Set α) J₂) := by
    intro hdis
    apply J₂S
    simp only [le_eq_subset, mem_setOf_eq, SetLike.coe_subset_coe, S]
    exact ⟨J₂.isIdeal, le_trans IJ' le_sup_left, hdis⟩

  -- Thus, pick cᵢ ∈ F ∩ Jᵢ.
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    a₂J₂ : Membership.mem J₂ a₂
    J₁J : Ne (↑J₁) Jset
    J₂J : Ne (↑J₂) Jset
    J₁S : Not (Membership.mem S ↑J₁)
    J₂S : Not (Membership.mem S ↑J₂)
    J₁F : Not (Disjoint ↑F ↑J₁)
    J₂F : Not (Disjoint ↑F ↑J₂)
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  let ⟨c₁, ⟨c₁F, c₁J₁⟩⟩ := Set.not_disjoint_iff.1 J₁F
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    a₂J₂ : Membership.mem J₂ a₂
    J₁J : Ne (↑J₁) Jset
    J₂J : Ne (↑J₂) Jset
    J₁S : Not (Membership.mem S ↑J₁)
    J₂S : Not (Membership.mem S ↑J₂)
    J₁F : Not (Disjoint ↑F ↑J₁)
    J₂F : Not (Disjoint ↑F ↑J₂)
    c₁ : α
    c₁F : Membership.mem (↑F) c₁
    c₁J₁ : Membership.mem (↑J₁) c₁
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  let ⟨c₂, ⟨c₂F, c₂J₂⟩⟩ := Set.not_disjoint_iff.1 J₂F

  -- Using the definition of Jᵢ, we can pick bᵢ ∈ J such that cᵢ ≤ bᵢ ⊔ aᵢ.
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    a₂J₂ : Membership.mem J₂ a₂
    J₁J : Ne (↑J₁) Jset
    J₂J : Ne (↑J₂) Jset
    J₁S : Not (Membership.mem S ↑J₁)
    J₂S : Not (Membership.mem S ↑J₂)
    J₁F : Not (Disjoint ↑F ↑J₁)
    J₂F : Not (Disjoint ↑F ↑J₂)
    c₁ : α
    c₁F : Membership.mem (↑F) c₁
    c₁J₁ : Membership.mem (↑J₁) c₁
    c₂ : α
    c₂F : Membership.mem (↑F) c₂
    c₂J₂ : Membership.mem (↑J₂) c₂
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  let ⟨b₁, ⟨b₁J, cba₁⟩⟩ := (mem_ideal_sup_principal a₁ c₁ J).1 c₁J₁
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    a₂J₂ : Membership.mem J₂ a₂
    J₁J : Ne (↑J₁) Jset
    J₂J : Ne (↑J₂) Jset
    J₁S : Not (Membership.mem S ↑J₁)
    J₂S : Not (Membership.mem S ↑J₂)
    J₁F : Not (Disjoint ↑F ↑J₁)
    J₂F : Not (Disjoint ↑F ↑J₂)
    c₁ : α
    c₁F : Membership.mem (↑F) c₁
    c₁J₁ : Membership.mem (↑J₁) c₁
    c₂ : α
    c₂F : Membership.mem (↑F) c₂
    c₂J₂ : Membership.mem (↑J₂) c₂
    b₁ : α
    b₁J : Membership.mem J b₁
    cba₁ : LE.le c₁ (Max.max b₁ a₁)
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  let ⟨b₂, ⟨b₂J, cba₂⟩⟩ := (mem_ideal_sup_principal a₂ c₂ J).1 c₂J₂

  -- Since J is an ideal, we have b := b₁ ⊔ b₂ ∈ J.
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    a₂J₂ : Membership.mem J₂ a₂
    J₁J : Ne (↑J₁) Jset
    J₂J : Ne (↑J₂) Jset
    J₁S : Not (Membership.mem S ↑J₁)
    J₂S : Not (Membership.mem S ↑J₂)
    J₁F : Not (Disjoint ↑F ↑J₁)
    J₂F : Not (Disjoint ↑F ↑J₂)
    c₁ : α
    c₁F : Membership.mem (↑F) c₁
    c₁J₁ : Membership.mem (↑J₁) c₁
    c₂ : α
    c₂F : Membership.mem (↑F) c₂
    c₂J₂ : Membership.mem (↑J₂) c₂
    b₁ : α
    b₁J : Membership.mem J b₁
    cba₁ : LE.le c₁ (Max.max b₁ a₁)
    b₂ : α
    b₂J : Membership.mem J b₂
    cba₂ : LE.le c₂ (Max.max b₂ a₂)
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  let b := b₁ ⊔ b₂
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    a₂J₂ : Membership.mem J₂ a₂
    J₁J : Ne (↑J₁) Jset
    J₂J : Ne (↑J₂) Jset
    J₁S : Not (Membership.mem S ↑J₁)
    J₂S : Not (Membership.mem S ↑J₂)
    J₁F : Not (Disjoint ↑F ↑J₁)
    J₂F : Not (Disjoint ↑F ↑J₂)
    c₁ : α
    c₁F : Membership.mem (↑F) c₁
    c₁J₁ : Membership.mem (↑J₁) c₁
    c₂ : α
    c₂F : Membership.mem (↑F) c₂
    c₂J₂ : Membership.mem (↑J₂) c₂
    b₁ : α
    b₁J : Membership.mem J b₁
    cba₁ : LE.le c₁ (Max.max b₁ a₁)
    b₂ : α
    b₂J : Membership.mem J b₂
    cba₂ : LE.le c₂ (Max.max b₂ a₂)
    b : α := Max.max b₁ b₂
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  have bJ : b ∈ J := sup_mem b₁J b₂J

  -- We now prove a key inequality, using crucially that the lattice is distributive.
  have ineq : c₁ ⊓ c₂ ≤ b ⊔ (a₁ ⊓ a₂) :=
  calc
    c₁ ⊓ c₂ ≤ (b₁ ⊔ a₁) ⊓ (b₂ ⊔ a₂) := inf_le_inf cba₁ cba₂
    _       ≤ (b  ⊔ a₁) ⊓ (b  ⊔ a₂) := by
      apply inf_le_inf <;> apply sup_le_sup_right; exact le_sup_left; exact le_sup_right
    _       = b ⊔ (a₁ ⊓ a₂) := (sup_inf_left b a₁ a₂).symm

  -- Note that c₁ ⊓ c₂ ∈ F, since c₁ and c₂ are both in F and F is a filter.
  -- Since F is an upper set, it now follows that b ⊔ (a₁ ⊓ a₂) ∈ F.
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    a₂J₂ : Membership.mem J₂ a₂
    J₁J : Ne (↑J₁) Jset
    J₂J : Ne (↑J₂) Jset
    J₁S : Not (Membership.mem S ↑J₁)
    J₂S : Not (Membership.mem S ↑J₂)
    J₁F : Not (Disjoint ↑F ↑J₁)
    J₂F : Not (Disjoint ↑F ↑J₂)
    c₁ : α
    c₁F : Membership.mem (↑F) c₁
    c₁J₁ : Membership.mem (↑J₁) c₁
    c₂ : α
    c₂F : Membership.mem (↑F) c₂
    c₂J₂ : Membership.mem (↑J₂) c₂
    b₁ : α
    b₁J : Membership.mem J b₁
    cba₁ : LE.le c₁ (Max.max b₁ a₁)
    b₂ : α
    b₂J : Membership.mem J b₂
    cba₂ : LE.le c₂ (Max.max b₂ a₂)
    b : α := Max.max b₁ b₂
    bJ : Membership.mem J b
    ineq : LE.le (Min.min c₁ c₂) (Max.max b (Min.min a₁ a₂))
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  have ba₁a₂F : b ⊔ (a₁ ⊓ a₂) ∈ F := PFilter.mem_of_le ineq (PFilter.inf_mem c₁F c₂F)

  -- Now, if we would have a₁ ⊓ a₂ ∈ J, then, since J is an ideal and b ∈ J, we would also get
  -- b ⊔ (a₁ ⊓ a₂) ∈ J. But this contradicts that J is disjoint from F.
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    JF : Disjoint (↑F) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    a₂J₂ : Membership.mem J₂ a₂
    J₁J : Ne (↑J₁) Jset
    J₂J : Ne (↑J₂) Jset
    J₁S : Not (Membership.mem S ↑J₁)
    J₂S : Not (Membership.mem S ↑J₂)
    J₁F : Not (Disjoint ↑F ↑J₁)
    J₂F : Not (Disjoint ↑F ↑J₂)
    c₁ : α
    c₁F : Membership.mem (↑F) c₁
    c₁J₁ : Membership.mem (↑J₁) c₁
    c₂ : α
    c₂F : Membership.mem (↑F) c₂
    c₂J₂ : Membership.mem (↑J₂) c₂
    b₁ : α
    b₁J : Membership.mem J b₁
    cba₁ : LE.le c₁ (Max.max b₁ a₁)
    b₂ : α
    b₂J : Membership.mem J b₂
    cba₂ : LE.le c₂ (Max.max b₂ a₂)
    b : α := Max.max b₁ b₂
    bJ : Membership.mem J b
    ineq : LE.le (Min.min c₁ c₂) (Max.max b (Min.min a₁ a₂))
    ba₁a₂F : Membership.mem F (Max.max b (Min.min a₁ a₂))
    ⊢ Not (Membership.mem J (Min.min a₁ a₂))
  -/
  contrapose! JF with ha₁a₂
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    a₂J₂ : Membership.mem J₂ a₂
    J₁J : Ne (↑J₁) Jset
    J₂J : Ne (↑J₂) Jset
    J₁S : Not (Membership.mem S ↑J₁)
    J₂S : Not (Membership.mem S ↑J₂)
    J₁F : Not (Disjoint ↑F ↑J₁)
    J₂F : Not (Disjoint ↑F ↑J₂)
    c₁ : α
    c₁F : Membership.mem (↑F) c₁
    c₁J₁ : Membership.mem (↑J₁) c₁
    c₂ : α
    c₂F : Membership.mem (↑F) c₂
    c₂J₂ : Membership.mem (↑J₂) c₂
    b₁ : α
    b₁J : Membership.mem J b₁
    cba₁ : LE.le c₁ (Max.max b₁ a₁)
    b₂ : α
    b₂J : Membership.mem J b₂
    cba₂ : LE.le c₂ (Max.max b₂ a₂)
    b : α := Max.max b₁ b₂
    bJ : Membership.mem J b
    ineq : LE.le (Min.min c₁ c₂) (Max.max b (Min.min a₁ a₂))
    ba₁a₂F : Membership.mem F (Max.max b (Min.min a₁ a₂))
    ha₁a₂ : Membership.mem J (Min.min a₁ a₂)
    ⊢ Not (Disjoint (↑F) Jset)
  -/
  rw [Set.not_disjoint_iff]
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    a₂J₂ : Membership.mem J₂ a₂
    J₁J : Ne (↑J₁) Jset
    J₂J : Ne (↑J₂) Jset
    J₁S : Not (Membership.mem S ↑J₁)
    J₂S : Not (Membership.mem S ↑J₂)
    J₁F : Not (Disjoint ↑F ↑J₁)
    J₂F : Not (Disjoint ↑F ↑J₂)
    c₁ : α
    c₁F : Membership.mem (↑F) c₁
    c₁J₁ : Membership.mem (↑J₁) c₁
    c₂ : α
    c₂F : Membership.mem (↑F) c₂
    c₂J₂ : Membership.mem (↑J₂) c₂
    b₁ : α
    b₁J : Membership.mem J b₁
    cba₁ : LE.le c₁ (Max.max b₁ a₁)
    b₂ : α
    b₂J : Membership.mem J b₂
    cba₂ : LE.le c₂ (Max.max b₂ a₂)
    b : α := Max.max b₁ b₂
    bJ : Membership.mem J b
    ineq : LE.le (Min.min c₁ c₂) (Max.max b (Min.min a₁ a₂))
    ba₁a₂F : Membership.mem F (Max.max b (Min.min a₁ a₂))
    ha₁a₂ : Membership.mem J (Min.min a₁ a₂)
    ⊢ Exists fun x => And (Membership.mem (↑F) x) (Membership.mem Jset x)
  -/
  use b ⊔ (a₁ ⊓ a₂)
  /-
    case h
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    F : Order.PFilter α
    I : Order.Ideal α
    hFI : Disjoint ↑F ↑I
    S : Set (Set α) := setOf fun J => And (Order.IsIdeal J) (And (LE.le (↑I) J) (D …
    Jset : Set α
    left✝ : HasSubset.Subset (↑I) Jset
    hmax : Maximal (fun x => Membership.mem S x) Jset
    Jidl : Order.IsIdeal Jset
    IJ : LE.le (↑I) Jset
    J : Order.Ideal α := Jidl.toIdeal
    IJ' : LE.le I J
    Jpr : J.IsProper
    a₁ a₂ : α
    ha₁ : Not (Membership.mem J a₁)
    ha₂ : Not (Membership.mem J a₂)
    J₁ : Order.Ideal α := Max.max J (Order.Ideal.principal a₁)
    J₂ : Order.Ideal α := Max.max J (Order.Ideal.principal a₂)
    a₁J₁ : Membership.mem J₁ a₁
    a₂J₂ : Membership.mem J₂ a₂
    J₁J : Ne (↑J₁) Jset
    J₂J : Ne (↑J₂) Jset
    J₁S : Not (Membership.mem S ↑J₁)
    J₂S : Not (Membership.mem S ↑J₂)
    J₁F : Not (Disjoint ↑F ↑J₁)
    J₂F : Not (Disjoint ↑F ↑J₂)
    c₁ : α
    c₁F : Membership.mem (↑F) c₁
    c₁J₁ : Membership.mem (↑J₁) c₁
    c₂ : α
    c₂F : Membership.mem (↑F) c₂
    c₂J₂ : Membership.mem (↑J₂) c₂
    b₁ : α
    b₁J : Membership.mem J b₁
    cba₁ : LE.le c₁ (Max.max b₁ a₁)
    b₂ : α
    b₂J : Membership.mem J b₂
    cba₂ : LE.le c₂ (Max.max b₂ a₂)
    b : α := Max.max b₁ b₂
    bJ : Membership.mem J b
    ineq : LE.le (Min.min c₁ c₂) (Max.max b (Min.min a₁ a₂))
    ba₁a₂F : Membership.mem F (Max.max b (Min.min a₁ a₂))
    ha₁a₂ : Membership.mem J (Min.min a₁ a₂)
    ⊢ And (Membership.mem (↑F) (Max.max b (Min.min a₁ a₂))) (Membership.mem Jset ( …
  -/
  exact ⟨ba₁a₂F, sup_mem bJ ha₁a₂⟩
  /-
    🎉 no goals
  -/

-- TODO: Define prime filters in Mathlib so that the following corollary can be stated and proved.
-- theorem prime_filter_of_disjoint_filter_ideal (hFI : Disjoint (F : Set α) (I : Set α)) :
--     ∃ G : PFilter α, (IsPrime G) ∧ F ≤ G ∧ Disjoint (G : Set α) I := by sorry


