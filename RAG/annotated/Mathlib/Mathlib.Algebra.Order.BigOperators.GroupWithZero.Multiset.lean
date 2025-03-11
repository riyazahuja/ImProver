lemma prod_nonneg {s : Multiset R} (h : ∀ a ∈ s, 0 ≤ a) : 0 ≤ s.prod := by
  /-
    R : Type u_1
    inst✝³ : CommMonoidWithZero R
    inst✝² : PartialOrder R
    inst✝¹ : ZeroLEOneClass R
    inst✝ : PosMulMono R
    s : Multiset R
    h : ∀ (a : R), Membership.mem s a → LE.le 0 a
    ⊢ LE.le 0 s.prod
  -/
  cases s using Quotient.ind
  /-
    case a
    R : Type u_1
    inst✝³ : CommMonoidWithZero R
    inst✝² : PartialOrder R
    inst✝¹ : ZeroLEOneClass R
    inst✝ : PosMulMono R
    a✝ : List R
    h : ∀ (a : R), Membership.mem (Quotient.mk (List.isSetoid R) a✝) a → LE.le 0 a
    ⊢ LE.le 0 (Multiset.prod (Quotient.mk (List.isSetoid R) a✝))
  -/
  simp only [quot_mk_to_coe, mem_coe, prod_coe] at *
  /-
    case a
    R : Type u_1
    inst✝³ : CommMonoidWithZero R
    inst✝² : PartialOrder R
    inst✝¹ : ZeroLEOneClass R
    inst✝ : PosMulMono R
    a✝ : List R
    h : ∀ (a : R), Membership.mem a✝ a → LE.le 0 a
    ⊢ LE.le 0 a✝.prod
  -/
  apply List.prod_nonneg h
  /-
    🎉 no goals
  -/


lemma one_le_prod {s : Multiset R} (h : ∀ a ∈ s, 1 ≤ a) : 1 ≤ s.prod := by
  /-
    R : Type u_1
    inst✝³ : CommMonoidWithZero R
    inst✝² : PartialOrder R
    inst✝¹ : ZeroLEOneClass R
    inst✝ : PosMulMono R
    s : Multiset R
    h : ∀ (a : R), Membership.mem s a → LE.le 1 a
    ⊢ LE.le 1 s.prod
  -/
  cases s using Quotient.ind
  /-
    case a
    R : Type u_1
    inst✝³ : CommMonoidWithZero R
    inst✝² : PartialOrder R
    inst✝¹ : ZeroLEOneClass R
    inst✝ : PosMulMono R
    a✝ : List R
    h : ∀ (a : R), Membership.mem (Quotient.mk (List.isSetoid R) a✝) a → LE.le 1 a
    ⊢ LE.le 1 (Multiset.prod (Quotient.mk (List.isSetoid R) a✝))
  -/
  simp only [quot_mk_to_coe, mem_coe, prod_coe] at *
  /-
    case a
    R : Type u_1
    inst✝³ : CommMonoidWithZero R
    inst✝² : PartialOrder R
    inst✝¹ : ZeroLEOneClass R
    inst✝ : PosMulMono R
    a✝ : List R
    h : ∀ (a : R), Membership.mem a✝ a → LE.le 1 a
    ⊢ LE.le 1 a✝.prod
  -/
  apply List.one_le_prod h
  /-
    🎉 no goals
  -/


theorem prod_map_le_prod_map₀ {ι : Type*} {s : Multiset ι} (f : ι → R) (g : ι → R)
    (h0 : ∀ i ∈ s, 0 ≤ f i) (h : ∀ i ∈ s, f i ≤ g i) :
    (map f s).prod ≤ (map g s).prod := by
  /-
    R : Type u_1
    inst✝³ : CommMonoidWithZero R
    inst✝² : PartialOrder R
    inst✝¹ : ZeroLEOneClass R
    inst✝ : PosMulMono R
    ι : Type u_2
    s : Multiset ι
    f g : ι → R
    h0 : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    h : ∀ (i : ι), Membership.mem s i → LE.le (f i) (g i)
    ⊢ LE.le (Multiset.map f s).prod (Multiset.map g s).prod
  -/
  cases s using Quotient.ind
  /-
    case a
    R : Type u_1
    inst✝³ : CommMonoidWithZero R
    inst✝² : PartialOrder R
    inst✝¹ : ZeroLEOneClass R
    inst✝ : PosMulMono R
    ι : Type u_2
    f g : ι → R
    a✝ : List ι
    h0 : ∀ (i : ι), Membership.mem (Quotient.mk (List.isSetoid ι) a✝) i → LE.le 0  …
    h : ∀ (i : ι), Membership.mem (Quotient.mk (List.isSetoid ι) a✝) i → LE.le (f  …
    ⊢ LE.le (Multiset.map f (Quotient.mk (List.isSetoid ι) a✝)).prod (Multiset.map …
  -/
  simp only [quot_mk_to_coe, mem_coe, map_coe, prod_coe] at *
  /-
    case a
    R : Type u_1
    inst✝³ : CommMonoidWithZero R
    inst✝² : PartialOrder R
    inst✝¹ : ZeroLEOneClass R
    inst✝ : PosMulMono R
    ι : Type u_2
    f g : ι → R
    a✝ : List ι
    h0 : ∀ (i : ι), Membership.mem a✝ i → LE.le 0 (f i)
    h : ∀ (i : ι), Membership.mem a✝ i → LE.le (f i) (g i)
    ⊢ LE.le (List.map f a✝).prod (List.map g a✝).prod
  -/
  apply List.prod_map_le_prod_map₀ f g h0 h
  /-
    🎉 no goals
  -/


lemma prod_pos {s : Multiset R} (h : ∀ a ∈ s, 0 < a) : 0 < s.prod := by
  /-
    R : Type u_1
    inst✝⁴ : CommMonoidWithZero R
    inst✝³ : PartialOrder R
    inst✝² : ZeroLEOneClass R
    inst✝¹ : PosMulStrictMono R
    inst✝ : NeZero 1
    s : Multiset R
    h : ∀ (a : R), Membership.mem s a → LT.lt 0 a
    ⊢ LT.lt 0 s.prod
  -/
  cases s using Quotient.ind
  /-
    case a
    R : Type u_1
    inst✝⁴ : CommMonoidWithZero R
    inst✝³ : PartialOrder R
    inst✝² : ZeroLEOneClass R
    inst✝¹ : PosMulStrictMono R
    inst✝ : NeZero 1
    a✝ : List R
    h : ∀ (a : R), Membership.mem (Quotient.mk (List.isSetoid R) a✝) a → LT.lt 0 a
    ⊢ LT.lt 0 (Multiset.prod (Quotient.mk (List.isSetoid R) a✝))
  -/
  simp only [quot_mk_to_coe, mem_coe, map_coe, prod_coe] at *
  /-
    case a
    R : Type u_1
    inst✝⁴ : CommMonoidWithZero R
    inst✝³ : PartialOrder R
    inst✝² : ZeroLEOneClass R
    inst✝¹ : PosMulStrictMono R
    inst✝ : NeZero 1
    a✝ : List R
    h : ∀ (a : R), Membership.mem a✝ a → LT.lt 0 a
    ⊢ LT.lt 0 a✝.prod
  -/
  apply List.prod_pos h
  /-
    🎉 no goals
  -/


theorem prod_map_lt_prod_map {ι : Type*} {s : Multiset ι} (hs : s ≠ 0)
    (f : ι → R) (g : ι → R) (h0 : ∀ i ∈ s, 0 < f i) (h : ∀ i ∈ s, f i < g i) :
    (map f s).prod < (map g s).prod := by
  /-
    R : Type u_1
    inst✝⁴ : CommMonoidWithZero R
    inst✝³ : PartialOrder R
    inst✝² : ZeroLEOneClass R
    inst✝¹ : PosMulStrictMono R
    inst✝ : NeZero 1
    ι : Type u_2
    s : Multiset ι
    hs : Ne s 0
    f g : ι → R
    h0 : ∀ (i : ι), Membership.mem s i → LT.lt 0 (f i)
    h : ∀ (i : ι), Membership.mem s i → LT.lt (f i) (g i)
    ⊢ LT.lt (Multiset.map f s).prod (Multiset.map g s).prod
  -/
  cases s using Quotient.ind
  /-
    case a
    R : Type u_1
    inst✝⁴ : CommMonoidWithZero R
    inst✝³ : PartialOrder R
    inst✝² : ZeroLEOneClass R
    inst✝¹ : PosMulStrictMono R
    inst✝ : NeZero 1
    ι : Type u_2
    f g : ι → R
    a✝ : List ι
    hs : Ne (Quotient.mk (List.isSetoid ι) a✝) 0
    h0 : ∀ (i : ι), Membership.mem (Quotient.mk (List.isSetoid ι) a✝) i → LT.lt 0  …
    h : ∀ (i : ι), Membership.mem (Quotient.mk (List.isSetoid ι) a✝) i → LT.lt (f  …
    ⊢ LT.lt (Multiset.map f (Quotient.mk (List.isSetoid ι) a✝)).prod (Multiset.map …
  -/
  simp only [quot_mk_to_coe, mem_coe, map_coe, prod_coe, ne_eq, coe_eq_zero] at *
  /-
    case a
    R : Type u_1
    inst✝⁴ : CommMonoidWithZero R
    inst✝³ : PartialOrder R
    inst✝² : ZeroLEOneClass R
    inst✝¹ : PosMulStrictMono R
    inst✝ : NeZero 1
    ι : Type u_2
    f g : ι → R
    a✝ : List ι
    hs : Not (Eq a✝ List.nil)
    h0 : ∀ (i : ι), Membership.mem a✝ i → LT.lt 0 (f i)
    h : ∀ (i : ι), Membership.mem a✝ i → LT.lt (f i) (g i)
    ⊢ LT.lt (List.map f a✝).prod (List.map g a✝).prod
  -/
  apply List.prod_map_lt_prod_map hs f g h0 h
  /-
    🎉 no goals
  -/


